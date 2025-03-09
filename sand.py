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

# penalty stiffness
kh = 1
# dynamic friction coefficient
dy = 0.1
n_rigid = 4
# number of seeding particles per rigid body
n_seg = 20
# rigid segments
x_r = ti.Vector.field(dim, float, (n_seg + 1, n_rigid))
# position of seeding particles
x_rp = ti.Vector.field(dim, float, (n_seg, n_rigid))
# velocity of seeding particles
v_rp = ti.Vector.field(dim, float, (n_seg, n_rigid))

# start point of fan
x_s = ti.Vector.field(dim, float, n_rigid)
# end point of fan
x_e = ti.Vector.field(dim, float, ())
m_fan = ti.field(float, ())
J_fan = ti.field(float, ())
# angular momentum
Mt = ti.field(float, ())
# angular velocity
omega = ti.field(float, ())

# distance to each rigid fan
grid_d = ti.Vector.field(n_rigid, float, (n_grid,) * dim)
# affinity
grid_A = ti.Vector.field(n_rigid, int, (n_grid,) * dim)
# tag
grid_T = ti.Vector.field(n_rigid, int, (n_grid,) * dim)
# index of closest rigid body
grid_r = ti.field(int, (n_grid,) * dim)
# index of closest rigid particle
grid_rp = ti.field(int, (n_grid,) * dim + (n_rigid,))

# particle distance to rigid fan
p_d = ti.Vector.field(n_rigid, float, max_particles)
# affinity
p_A = ti.Vector.field(n_rigid, int, max_particles)
# tag
p_T = ti.Vector.field(n_rigid, int, max_particles)
# normal
p_n = ti.Vector.field(dim, float, (max_particles, n_rigid))

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
def calculate_CDF():
    for i, j in grid_A:
        for k in ti.static(range(n_rigid)):
            grid_A[i, j][k] = 0
            grid_T[i, j][k] = 0
            grid_d[i, j][k] = -1.0
            grid_rp[i, j, k] = -1
        grid_r[i, j] = -1
        
    for k in ti.static(range(n_rigid)):
        for p in range(n_seg):
            ba = x_r[p + 1, k] - x_r[p, k]
            base = (x_rp[p, k] * inv_dx - 0.5).cast(int)
            # if base[0] < 0 or base[0] >= n_grid or base[1] < 0 or base[1] >= n_grid:
            #     print(p, k, base, x_r[p + 1, k], x_r[p, k])
            #     assert False, "WRONG!!"
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
                pa = (offset + base).cast(float) * dx - x_r[p, k]
                h = pa.dot(ba) / ba.dot(ba)
                
                if h >= 0.0 and h <= 1.0:
                    temp = base + offset
                    grid_d[temp][k] = (pa - h * ba).norm()
                    grid_A[temp][k] = 1
                    grid_rp[temp[0], temp[1], k] = p
                    cross = pa[0] * ba[1] - pa[1] * ba[0]
                    grid_T[temp][k] = 1 if cross > 0.0 else -1

    for i, j in grid_r:
        d_min = 1e7
        for k in ti.static(range(n_rigid)):
            if grid_A[i, j][k] == 1 and grid_d[i, j][k] < d_min:
                d_min = grid_d[i, j][k]
                grid_r[i, j] = k
    for k in ti.static(range(n_rigid)):
        for p in range(n_particles[None]):
            p_A[p][k] = 0
            p_T[p][k] = 0
            p_d[p][k] = 0.0
            
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            Tpr = 0.0
            
            d_vecs = ti.Vector.zero(float, 9)
            diag = ti.Matrix.identity(float, 9)
            Q = ti.Matrix.zero(float, 9, 3)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
                if grid_A[base + offset][k] == 1:
                    p_A[p][k] = 1
                d_sign = grid_T[base + offset][k] * grid_d[base + offset][k]
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                dpos = (offset.cast(float) - fx) * dx
                
                n = offset[0] * 3 + offset[1]
                d_vecs[n] = d_sign
                diag[n, n] = weight
                Q[n, 0] = 1
                Q[n, 1] = dpos[0]
                Q[n, 2] = dpos[1]
                
                Tpr += weight * d_sign
                
            if p_A[p][k] == 1:
                if p_T[p][k] == 0:
                    p_T[p][k] = 1 if Tpr > 0.0 else -1
                M = Q.transpose() @ diag @ Q
                dist_p = M.inverse() @ Q.transpose() @ diag @ d_vecs
                p_d[p][k] = dist_p[0]
                p_n[p, k] = ti.Vector([dist_p[1], dist_p[2]]).normalized()
            else:
                p_T[p][k] = 0
                
@ti.kernel
def P2G():
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
            flag = True
            for k in ti.static(range(n_rigid)):
                if p_T[p][k] == grid_T[base + offset][k] or p_T[p][k] * grid_T[base + offset][k] == 0:
                    pass
                else:
                    flag = False
            if flag == True:
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
            grid_f[base + offset] += weight * stress @ dpos
            grid_mg[base + offset] += -p_mass * grad_weight

@ti.kernel
def apply_BC():        
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

@ti.kernel
def G2P():
    Mt[None] = 0.0
    # G2P
    for p in range(n_particles[None]):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        new_rho = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = ti.Vector.zero(float, dim)
            flag = True
            for k in ti.static(range(n_rigid)):
                if p_T[p][k] == grid_T[base + offset][k] or p_T[p][k] * grid_T[base + offset][k] == 0:
                    pass
                else:
                    flag = False
            if flag == False:
                temp = base + offset
                r_body = grid_r[temp]
                r_id = grid_rp[temp[0], temp[1], r_body]
                line = (x_r[r_id + 1, r_body] - x_r[r_id, r_body]).normalized()
                pa = x[p] - x_r[r_id, r_body]
                np = (pa - pa.dot(line) * line).normalized()
                sg = (v[p] - v_rp[r_id, r_body]).dot(np)
                if sg > 0:
                    g_v = v[p]
                else:
                    vt = v[p] - v_rp[r_id, r_body] - sg * np
                    xi = max(0, vt.norm() + dy * sg)
                    g_v = vt.normalized() * xi + v_rp[r_id, r_body]
                    
                    # angualr momentum change
                    rp = x_rp[r_id, r_body] - x_r[n_seg, r_body]
                    mvp = p_mass * weight * (v[p] - g_v)
                    Mt[None] += rp[0] * mvp[1] - rp[1] * mvp[0]
            else:
                g_v = grid_v[base + offset]
                    
            dpos = (offset.cast(float) - fx) * dx
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx**2
            new_rho += weight * (grid_m[base + offset] - grid_mg[base + offset].dot(dpos))
        
        v[p], C[p] = new_v, new_C
        # APIC update
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        
        for k in ti.static(range(n_rigid)):
            if p_T[p][k] * p_d[p][k] < 0:
                f_penalty = kh * p_d[p][k] * p_n[p, k]
                v[p] += dt * f_penalty / p_mass
        
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
def rigid_advection():
    dw = Mt[None] / J_fan[None]
    omega[None] += dw
    og_Vec = ti.Vector([0.0, 0.0, omega[None]])
    for p, body in x_rp:
        rp = x_rp[p, body] - x_r[n_seg, body]
        rp_Vec = ti.Vector([rp[0], rp[1], 0.0])
        vrp = og_Vec.cross(rp_Vec)
        v_rp[p, body] = ti.Vector([vrp[0], vrp[1]])
        x_rp[p, body] += dt * v_rp[p, body]
    
    for j in ti.static(range(n_rigid)):
        for i in range(n_seg - 1):
            x_r[i + 1, j] = (x_rp[i, j] + x_rp[i + 1, j]) * 0.5
        x_r[0, j] = 2 * x_rp[0, j] - x_r[1, j]
        x_r[n_seg, j] = 2 * x_rp[n_seg - 1, j] - x_r[n_seg - 1, j]

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
            x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.7]
        v[i] = ti.Vector.zero(float, dim)
        F[i] = ti.Matrix.identity(float, dim)
        color[i] = ti.Vector([210 / 255, 170 / 255, 109 / 255, 1])
        alpha[i] = 0.067765
    
    x_e[None] = [0.5, 0.5] if dim == 2 else [0.5, 0.5, 0.5]
    length = 0.1
    m_fan[None] = 2
    J_fan[None] = m_fan[None] * length ** 2 / 3.0 * n_rigid
    omega[None] = -15
    for i in range(n_rigid):
        x_s[i] = [x_e[None][0] + length * ti.cos(i / n_rigid * math.pi * 2), \
                  x_e[None][1] + length * ti.sin(i / n_rigid * math.pi * 2)]
        x_r[0, i] = x_s[i]
        for j in range(n_seg):
            x_r[j + 1, i] = x_s[i] + (x_e[None] - x_s[i]) / n_seg * (j + 1)
            x_rp[j, i] = (x_r[j, i] + x_r[j + 1, i]) * 0.5
    
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

def substep():
    calculate_CDF()
    P2G()
    apply_BC()
    G2P()
    rigid_advection()

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
            for i in range(n_rigid):
                gui.line(x_r.to_numpy()[0, i], x_r.to_numpy()[-1, i], radius = 1.5, color = 0x068587)
            # for i in range(n_rigid):
            #     rp_positions = x_rp.to_numpy()[:, i]
            #     gui.circles(rp_positions, radius=.5, color=0xFF5733)
            gui.show(f'res/{frame:06d}.png' if write_to_disk else None)

if __name__ == "__main__":
    main()