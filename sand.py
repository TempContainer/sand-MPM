import taichi as ti
import numpy as np
import math
import os

ti.init(arch = ti.gpu)

# change this to determine whether write to disk
write_to_disk = False
if write_to_disk:
    os.mkdir('res')
    
# change this to change dimension
dim = 2
    
quality = 1
max_particles = 20000 * quality ** 2
n_particles = ti.field(int, ())
n_grid = 128 * quality
padding = 3
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-4 / quality

gravity = ti.Vector.field(dim, float, ())
gravity[None] = [0, 10, 0] if dim == 3 else [0, 10]
x = ti.Vector.field(dim, float, max_particles)
v = ti.Vector.field(dim, float, max_particles)
C = ti.Matrix.field(dim, dim, float, max_particles)
F = ti.Matrix.field(dim, dim, float, max_particles)
alpha = ti.field(float, max_particles)
# volume correction
vc = ti.field(float, max_particles)
q = ti.field(float, max_particles)
# sediment density
rho_hat = ti.field(float, ())

color = ti.Vector.field(4, float, max_particles)

grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
grid_m = ti.field(float, (n_grid,) * dim)
grid_f = ti.Vector.field(dim, float, (n_grid,) * dim)
# mass gradient
grid_mg = ti.Vector.field(dim, float, (n_grid,) * dim)

p_vol, p_rho = (dx * 0.5) ** 2, 400
p_mass = p_vol * p_rho

E, nu = 3.537e5, 0.3
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
h0, h1, h2, h3 = 35, 9, 0.2, 10
mu_b = 0.75

@ti.func
def log_mat(mat):
    res = ti.zero(mat)
    for i in ti.static(range(dim)):
        res[i, i] = ti.log(mat[i, i])
    return res

@ti.func
def exp_mat(mat):
    res = ti.zero(mat)
    for i in ti.static(range(dim)):
        res[i, i] = ti.exp(mat[i, i])
    return res

@ti.func
def F_norm(mat):
    norm = 0.0
    for i in ti.static(range(dim)):
        norm += mat[i, i] ** 2
    return ti.sqrt(norm)

# see Drucker-Prager Elastoplasticity for Sand Animation: Supplementary Technical Document
@ti.func
def project(sig, p):
    # volume correction 1
    eps = log_mat(sig) + vc[p] / dim * ti.Matrix.identity(float, dim)
    eps_hat = eps - eps.trace() / dim * ti.Matrix.identity(float, dim)
    eps_Fnorm, eps_hat_Fnorm = F_norm(eps), F_norm(eps_hat)
    delta_gamma = eps_hat_Fnorm + (dim * lambda_0 / (2 * mu_0) + 1) * eps.trace() * alpha[p]
    H = eps - delta_gamma * eps_hat / eps_hat_Fnorm
    res_m = ti.Matrix.identity(float, dim)
    res_n = 0.0
    if eps_hat_Fnorm == 0.0 or eps.trace() > 0.0:
        res_n = eps_Fnorm
    elif delta_gamma <= 0.0:
        res_m, res_n = sig, 0.0
    else:
        res_m, res_n = exp_mat(H), delta_gamma
    return res_m, res_n

@ti.kernel
def substep():
    # clear grid
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_f[I] = ti.zero(grid_f[I])
        grid_mg[I] = ti.zero(grid_mg[I])
        grid_m[I] = 0.0
    
    # P2G
    # base framework brought from MLS-MPM
    for p in range(n_particles[None]):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        U, sig, V = ti.svd(F[p])
        inv_sig = sig.inverse()
        log_sig = log_mat(sig)
        stress = U @ (2 * mu_0 * inv_sig * log_sig + lambda_0 * log_sig.trace() * inv_sig) @ V.transpose()
        stress = (-p_vol * 4 * inv_dx**2) * stress @ F[p].transpose()
        affine = p_mass * C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            grad_weight = ti.Vector.one(float, dim) * inv_dx
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
                for j in ti.static(range(dim)):
                    if j == i:
                        grad_weight[i] *= grad_w[offset[i]][i]
                    else:
                        grad_weight[i] *= w[offset[i]][i]
                        
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
            grid_f[base + offset] += weight * stress @ dpos
            grid_mg[base + offset] += -p_mass * grad_weight
        
    # boundary conditions
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] += dt * grid_f[I]
            grid_v[I] /= grid_m[I]
        grid_v[I] += dt * gravity[None]
        delta = 0.0
        v_norm = grid_v[I].norm()
        # apply friction
        for d in ti.static(range(dim)):
            if I[d] < padding and grid_v[I][d] < 0:
                delta += grid_v[I][d] ** 2
                grid_v[I][d] = 0
            if I[d] > n_grid - padding and grid_v[I][d] > 0:
                delta += grid_v[I][d] ** 2
                grid_v[I][d] = 0
        delta = ti.sqrt(delta)
        grid_v[I] *= max(0, 1 - mu_b * delta / v_norm)
    
    # G2P
    for p in range(n_particles[None]):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        new_rho = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx**2
            new_rho += weight * (grid_m[base + offset] - grid_mg[base + offset].dot(dpos))
        
        v[p], C[p] = new_v, new_C
        # APIC update
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        x[p] += dt * v[p]
        
        U, sig, V = ti.svd(F[p])
        T, delta_q = project(sig, p)
        new_F = U @ T @ V.transpose()
        vc[p] += ti.log(F[p].determinant()) - ti.log(new_F.determinant())
        # volume correction 2
        F[p] = ti.Matrix.identity(float, dim) if new_rho < 0.8 * rho_hat[None] else new_F
        q[p] += delta_q
        phi = h0 + (h1 * q[p] - h3) * ti.exp(-h2 * q[p])
        sin_phi = ti.sin(phi / 180 * math.pi)
        alpha[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

@ti.kernel
def update_gravity():
    gravity[None] *= -1

@ti.kernel
def initialize():
    n_particles[None] = 10000 * quality ** 2
    for i in range(n_particles[None]):
        if ti.static(dim == 3):
            x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.5, ti.random() * 0.2 + 0.4]
        else:
            x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.5]
        v[i] = ti.Vector.zero(float, dim)
        F[i] = ti.Matrix.identity(float, dim)
        color[i] = ti.Vector([210 / 255, 170 / 255, 109 / 255, 1])
        alpha[i] = 0.067765
    # before starting, run once to get the average density
    for p in range(n_particles[None]):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            weight = 1.0
            grad_weight = ti.Vector.one(float, dim) * inv_dx
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
                for j in ti.static(range(dim)):
                    if j == i:
                        grad_weight[i] *= grad_w[offset[i]][i]
                    else:
                        grad_weight[i] *= w[offset[i]][i]
            grid_m[base + offset] += weight * p_mass
            grid_mg[base + offset] += -p_mass * grad_weight
    for p in range(n_particles[None]):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_rho = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            new_rho += weight * (grid_m[base + offset] - grid_mg[base + offset].dot(dpos))
        rho_hat[None] += new_rho
    rho_hat[None] /= n_particles[None]

def main():
    initialize()
    if dim == 3:
        res = (720, 720)
        window = ti.ui.Window("Sand", res, vsync=True)
        canvas = window.get_canvas()
        gui = window.get_gui()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(0.5, 1.0, 1.95)
        camera.lookat(0.5, 0.3, 0.5)
        camera.fov(55)

        def render():
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0, 0, 0))
            scene.particles(x, per_vertex_color=color, radius=0.005)
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
            canvas.scene(scene)
            
        for frame in range(1200 + 120):
            if frame % 120 == 0:
                update_gravity()
            for _ in range(50):
                substep()
            render()
            if write_to_disk:
                window.save_image(f'res/{frame:06d}.png' if write_to_disk else None)
            else:
                window.show()
    else:
        gui = ti.GUI("Sand", res = 512, background_color = 0x112F41)

        for frame in range(1200 + 120):
            if frame % 120 == 0:
                update_gravity()
            for _ in range(50):
                substep()
            gui.circles(x.to_numpy(), radius = 1.5, color = 0xD2AA6D)
            gui.show(f'res/{frame:06d}.png' if write_to_disk else None)

if __name__ == "__main__":
    main()