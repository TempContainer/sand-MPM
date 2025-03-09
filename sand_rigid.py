import taichi as ti
import numpy as np
import math
import os

ti.init(arch = ti.gpu)

# change this to determine whether write to disk
write_to_disk = False
if write_to_disk and not os.path.exists('res'):
    os.mkdir('res')
    
# change this to change dimension
dim = 3
    
quality = 1
max_particles = 10000 * quality ** 2
n_particles = ti.field(int, ())
n_grid = 128 * quality
padding = 3
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-4 / quality

gravity = ti.Vector.field(dim, float, ())
gravity[None] = [0, -10, 0] if dim == 3 else [0, -10]
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

# mass of the capsule
m_cpsl = 20
r = 0.05
# position of the capsule
x_c = ti.Vector.field(dim, float, 3)
# quaternion of the capsule
q_c = ti.Vector.field(4, float, ())
# rotation matrix of the capsule
R_c = ti.Matrix.field(dim, dim, float, ())
# inertia tensor of the capsule
I_c = ti.Matrix.field(dim, dim, float, ())
v_c = ti.Vector.field(dim, float, ())
omega_c = ti.Vector.field(dim, float, ())
J_c = ti.Vector.field(dim, float, ())
# accumulated torque
tau_c = ti.Vector.field(dim, float, ())


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

@ti.func
def normal_capsule(p):
    pa, ba = p - x_c[1], x_c[2] - x_c[1]
    h = ti.math.clamp(pa.dot(ba) / ba.dot(ba), 0.0, 1.0)
    return pa - ba * h

@ti.func
def sd_capsule(p):
    return normal_capsule(p).norm() - r

@ti.func
def quat_mul(q1, q2):
    return ti.math.vec4(
        q1.x * q2.x - ti.math.dot(q1.yzw, q2.yzw),
        q1.x * q2.yzw + q2.x * q1.yzw + ti.math.cross(q1.yzw, q2.yzw)
    )

@ti.func
def to_rot(q):
    return ti.Matrix([
        [1 - 2 * q.z**2 - 2 * q.w**2, 2 * q.y * q.z - 2 * q.w * q.x, 2 * q.y * q.w + 2 * q.z * q.x],
        [2 * q.y * q.z + 2 * q.w * q.x, 1 - 2 * q.y**2 - 2 * q.w**2, 2 * q.z * q.w - 2 * q.y * q.x],
        [2 * q.y * q.w - 2 * q.z * q.x, 2 * q.z * q.w + 2 * q.y * q.x, 1 - 2 * q.y**2 - 2 * q.z**2]
    ])

@ti.func
def to_mat(v):
    return ti.Matrix([
        [0, -v.z, v.y],
        [v.z, 0, -v.x],
        [-v.y, v.x, 0]
    ])

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
    
    J_c[None] = ti.Vector.zero(float, dim)
    tau_c[None] = ti.Vector.zero(float, dim)
    v_c[None] += dt * gravity[None]
    I_c_inv = (R_c[None] @ I_c[None] @ R_c[None].transpose()).inverse()
    
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
        grid_v[I] *= max(0, 1 - mu_b * ti.sqrt(delta) / v_norm)
        
        # collision
        x_i = I.cast(float) * dx
        d = sd_capsule(x_i)
        if d < 1e-5:
            x_i_hat = x_i + dt * grid_v[I]
            d_hat = sd_capsule(x_i_hat) - min(d, 1e-5)
            n = normal_capsule(x_i_hat).normalized()
            dv = d_hat * n / dt
            grid_v[I] -= dv
            dJ = dv * grid_m[I]
            J_c[None] += dJ
            tau_c[None] += (x_i - x_c[0]).cross(dJ)
    
    x_i = x_c[1] if x_c[1][1] < x_c[2][1] else x_c[2]
    x_i[1] -= r
    if x_i[1] < 1e-4:
        n = ti.Vector([0, 1, 0])
        x_r = x_i - x_c[0]
        v_i = v_c[None] + omega_c[None].cross(x_r)
        v_in = v_i.dot(n) * n
        if v_i.dot(n) < 1e-4:
            v_it = v_i - v_in
            a = max(1 - 0.9 * (1 + 0.5) * v_in.norm() / v_it.norm(), 0.0)
            v_ii = -0.5 * v_in + a * v_it
            x_rc = to_mat(x_r)
            K = ti.Matrix.diag(dim, 1.0 / m_cpsl) - x_rc @ I_c_inv @ x_rc
            dJ = K.inverse() @ (v_ii - v_i)
            J_c[None] += dJ
            tau_c[None] += x_r.cross(dJ)
    
    v_c[None] += J_c[None] / m_cpsl
    omega_c[None] += I_c_inv @ tau_c[None]
    
    q_c[None] = (q_c[None] + quat_mul(ti.math.vec4(0, 0.5 * dt * omega_c[None]), q_c[None])).normalized()
    R_c[None] = to_rot(q_c[None])
    x_c[0] += dt * v_c[None]
    x_c[1] = x_c[0] + R_c[None] @ ti.Vector([0, 0, -r])
    x_c[2] = x_c[0] + R_c[None] @ ti.Vector([0, 0, r])
    
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
def initialize():
    if ti.static(dim == 3):
        n_particles[None] = max_particles
    else:
        n_particles[None] = 10000 * quality ** 2
    for i in range(n_particles[None]):
        if ti.static(dim == 3):
            x[i] = [ti.random() * 0.8 + 0.1, ti.random() * 0.2 + 0.75, ti.random() * 0.8 + 0.1]
        else:
            x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.5]
        v[i] = ti.Vector.zero(float, dim)
        F[i] = ti.Matrix.identity(float, dim)
        color[i] = ti.Vector([210 / 255, 170 / 255, 109 / 255, 1])
        alpha[i] = 0.067765
    x_c[1], x_c[2] = [0.5, 0.4, 0.4], [0.5, 0.4, 0.4 + 2 * r]
    x_c[0] = (x_c[1] + x_c[2]) * 0.5
    q_c[None] = ti.Vector([1, 0, 0, 0])
    R_c[None] = ti.Matrix.identity(float, dim)
    I_c[None] = ti.Matrix.diag(dim, m_cpsl * r**2 / 3.0)
    I_c[None][2, 2] = 0.46 * m_cpsl * r**2
    
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
            scene.particles(x_c, radius=r)
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            canvas.scene(scene)
            
        for frame in range(1200 + 120):
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
            for _ in range(50):
                substep()
            gui.circles(x.to_numpy(), radius = 1.5, color = 0xD2AA6D)
            gui.show(f'res/{frame:06d}.png' if write_to_disk else None)

if __name__ == "__main__":
    main()