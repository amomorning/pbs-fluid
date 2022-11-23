import taichi as ti

ti.init(arch=ti.cuda)

# Control flag
wind_on = False
MAC_on = False
velocity_on = False

# Discretization parameter
res_x = 128
res_y = int(res_x * 1.5)
dx = 1.0
dt = 0.005 * ti.sqrt((res_x + res_y) * 0.5)
acc = 1e-5  # accuracy
n_iters = 1000
r = 0.0 # Parameter for wind force

# Quantities
    # Simple grid
density = ti.field(float, shape=(res_x, res_y))
pressure = ti.field(float, shape=(res_x, res_y))
divergence = ti.field(float, shape=(res_x, res_y))
vorticity = ti.field(float, shape=(res_x, res_y))

    # MAC grid
u = ti.field(float, shape=(res_x+1, res_y))
v = ti.field(float, shape=(res_x, res_y+1))
f_x = ti.field(float, shape=(res_x+1, res_y))
f_y = ti.field(float, shape=(res_x, res_y+1))

# For rendering
num_vertices = (res_x+1) * (res_y+1)
num_triangles = 2*res_x*res_y
V = ti.Vector.field(3, dtype=float, shape =num_vertices)
F = ti.field(int, shape = num_triangles * 3)

# Apply source
@ti.kernel
def apply_source():
    xmin = int(0.45 * res_x)
    xmax = int(0.55 * res_y)
    ymin = int(0.10 * res_y)
    ymax = int(0.15 * res_y)
    for i, j in ti.ndrange(xmax-xmin, ymax-ymin):
        density[i+xmin, j+ymin] = 1

# Advection-step functions
# Semi Lagrangian
@ti.kernel
def advect_quantity(q: ti.template(), u: ti.template(), v: ti.template(), dt: float, dx: float):
    # new values stored in q_tmp
    q_tmp = ti.field(float, shape=q.shape)

    for y in range(1, q.shape[1]):
        for x in range(1, q.shape[0]):
            # Velocity on grid point by lerp
            last_x_velocity = 0.5 * (u[x, y] + u[x+1, y])
            last_y_velocity = 0.5 * (v[x, y] + v[x+1, y])

            # Last position of the particle (in grid coordinates, that's why divided by dx)
            last_x = x - dt / dx * last_x_velocity
            last_y = y - dt / dx * last_y_velocity

            # Clamping
            if last_x < 1: last_x = 1
            if last_y < 1: last_y = 1
            if last_x > q.shape[0] - 2: last_x = q.shape[0] - 2
            if last_y > q.shape[1] - 2: last_y = q.shape[1] - 2

            # Corners for bilinear interpolation
            x_low = int(last_x)
            y_low = int(last_y)
            x_high = x_low + 1
            y_high = y_low + 1
            corners = [q[x_low, y_low], q[x_high, y_low], q[x_low, y_high], q[x_high, x_high]]

            # Bilinear interpolation weights
            x_weight = last_x - x_low
            y_weight = last_y - y_low
            weights = [x_weight, y_weight]

            q_tmp[x, y] = bilerp(weights, corners)

    q = q_tmp

@ti.func
def bilerp(weights, corners):
    """
    Calculate bilinear interpolation given weights and corners:
    
    Parameters:
    weights: [x_weight, y_weight]
    corners: [bot_left, bot_right, top_left, top_right]

    Return
    Bilinear interpolation of the square
    """
    result = (1 - weights[0]) * (1 - weights[1]) * corners[0]
    ... + weights[0] * (1 - weights[1]) * corners[1]
    ... + (1 - weights[0]) * weights[0] * corners[2]
    ... + weights[0] * weights[1] * corners[3]

    return result

@ti.kernel
def advect_velocity(u: ti.template(), v: ti.template(), dt: float, dx: float):
    u_tmp = ti.field(float, shape=u.shape)
    v_tmp = ti.field(float, shape=v.shape)

    res_x = v.shape[0]
    res_y = u.shape[1]

    # Advect u
    for y in range(1, u.shape[1]-1):
        for x in range(1, u.shape[0] - 1):
            # Velocity at MAC grid point
            last_x_velocity = u[x, y]
            last_y_velocity = (v[x, y] + v[x-1, y] + v[x-1, y+1] + v[x, y+1]) / 4

            # Last position (in grid coordinates)
            last_x = x - last_x_velocity * dt / dx
            last_y = y - last_y_velocity * dt / dx

            # Clamping
            if last_x < 1.5: last_x = 1.5
            if last_y < 1.5: last_y = 1.5
            if last_x > res_x - 1.5: last_x = res_x - 1.5
            if last_y > res_y - 2.5: last_y = res_y - 2.5

            # Corners for bilinear interpolation
            x_low = int(last_x)
            y_low = int(last_y)
            x_high = x_low + 1
            y_high = y_low + 1
            corners = [u[x_low, y_low], u[x_high, y_low], u[x_low, y_high], u[x_high, x_high]]

            # Bilinear interpolation weights
            x_weight = last_x - x_low
            y_weight = last_y - y_low
            weights = [x_weight, y_weight]

            u_tmp = bilerp(weights, corners)

    # Advect v
    for y in range(1, v.shape[1] - 1):
        for x in range(1, v.shape[0] - 1):
            # Velocity at MAC grid point
            last_x_velocity = (u[x, y] + u[x+1, y] + u[x+1, y-1] + u[x, y-1]) / 4
            last_y_velocity = v[x,y]

            # Clamping
            if last_x < 1.5: last_x = 1.5
            if last_y < 1.5: last_y = 1.5
            if last_x > res_x - 2.5: last_x = res_x - 2.5
            if last_y > res_y - 1.5: last_y = res_y - 1.5

            # Corners for bilinear interpolation
            x_low = int(last_x)
            y_low = int(last_y)
            x_high = x_low + 1
            y_high = y_low + 1
            corners = [v[x_low, y_low], v[x_high, y_low], v[x_low, y_high], v[x_high, x_high]]

            # Bilinear interpolation weights
            x_weight = last_x - x_low
            y_weight = last_y - y_low
            weights = [x_weight, y_weight]

            v_tmp = bilerp(weights, corners)

    u = u_tmp
    v = v_tmp

# Update-after-force functions
@ti.kernel
def add_buoyancy(f_y: ti.template()): # Vertical buoyancy
    scaling = 64.0 / f_y.shape[0]

    for i in range(f_y.shape[0]):
        for j in range(1, f_y.shape[1]-1):
            f_y[i,j] += 0.1 * (density[i, j-1] + density[i,j]) / 2 * scaling

@ti.kernel
def add_wind(f_x: ti.template(), t_curr: float, dt: float): # Horizontal wind
    scaling = 64.0 / f_x.shape[1]

    r = t_curr // dt

    f = 2e-2 * ti.cos(5e-2 * r) * ti.cos(3e-2 * r) * scaling

    for i, j in f_x:
        f_x[i,j] += f
    

@ti.kernel
def apply_force(u: ti.template(), v: ti.template(), f_x:ti.template(), f_y: ti.template()):
    for i,j in u:
        u[i,j] += dt * f_x[i,j]

    for i,j in v:
        v[i,j] += dt * f_y[i,j]

# Projection step
@ti.kernel
def set_Neuman(u: ti.template(), v: ti.template()): # velocity boundary condition 
    # ???????? Problem here, seems u, v not right placed
    """
    Velocity boundary condition

    For u 
    |[0]|  |[1]|  |[2]|       |[border-3]| |[border-2]| |[border-1]|
         \\     //                  \\                   //
          \\   //                    \\                 //
           \\ //                      \\===============//

    For v 

                    |[border-1]| \\ 
                    |[border-2]| ||
                    |[border-3]| // 

                    |[2]| \\ 
                    |[1]| ||
                    |[0]| // 
    
    """
    sx = u.shape[0]
    sy = u.shape[1]
    for y in range(sy):
        u[0,y] = u[2,y]
        u[sx-1, y] = u[sx-3, y]

    sx = v.shape[0]
    sy = v.shape[1]
    for x in range(sx):
        v[x,0] = v[x,2]
        v[x, sy-1] = v[x, sy-3]

@ti.kernel
def set_zero(u: ti.template(), v: ti.template()): # velocity boundary condition
    # ???????? Problem here, seems u, v not right placed
    """
    Velocity boundary condition
    u at ? is zero
    v at ? is zero
    """
    sx = u.shape[0]
    sy = u.shape[1]
    for x in range(sx):
        u[x,0] = 0
        u[x, sy-1] = 0

    sx = v.shape[0]
    sy = v.shape[1]
    for y in range(sy):
        v[0,y] = 0
        v[sy-1, y] = 0

@ti.kernel
def copy_boder(pressure: ti.template()): # 
    """
    Pressure boundary condition:
    Copy inner boder value to border

                    |[border-1]|
                        /\ 
                        ||
                    |[border-2]|

    |[0]| <= |[1]|                |[border-2]| => |[border-1]|

                      |[ 1 ]|
                        ||
                        \/
                      |[ 0 ]|
    """
    sx = pressure.shape[0]
    sy = pressure.shape[1]
    for y in range(sy):
        pressure[0,y] = pressure[1,y]
        pressure[sx-1, y] = pressure[sx-2, y]

    for x in range(sx):
        pressure[x,0] = pressure[x,1]
        pressure[x,  sy-1] = pressure[x, sy-2]

@ti.kernel
def solve_poisson(pressure: ti.template(), divergence: ti.template(), acc: float, dx: float, n_iters: int):
    dx2 = dx * dx
    residual = acc + 1
    rho = 1 
    it = 0

    while residual > acc and it < n_iters:
        for y in range(1, pressure.shape[1]-1):
            for x in range(1, pressure.shape[0]-1):
                b = -divergence[x,y] / dt * rho
                pressure[x,y] = (dx2 * b + pressure[x-1, y] + pressure[x+1, y] + pressure[x, y-1] + pressure[x, y+1]) / 4
        
        residual = 0
        for y in range(1, pressure.shape[1]-1):
            for x in range(1, pressure.shape[0]-1):
                b = -divergence[x,y] / dt * rho
                cell_residual = b - (4 * pressure[x, y] - pressure[x-1, y] - pressure[x+1, y] - pressure[x, y-1] - pressure[x, y+1]) / dx2 
                residual += cell_residual ** 2

        residual = ti.sqrt(residual)
        residual /= (pressure.shape[0] - 2) * (pressure.shape[1] - 2)

        it += 1

@ti.kernel
def correct_velocity(pressure: ti.template(), u: ti.template(), v: ti.template(), dt: float, dx: float):
    rho = 1

    for y in range(1, u.shape[1]-1):
        for x in range(0, u.shape[0]-1):
            u[x, y] = u[x, y] - dt / rho * (pressure[x, y] - pressure[x-1, y]) / dx

    for y in range(1, v.shape[1] - 1):
        for x in range(1, v.shape[0] - 1):
            v[x, y] = v[x, y] - dt / rho * (pressure[x, y] - pressure[x, y-1]) / dx


# Calculation of other quantities
@ti.kernel
def compute_divergence(divergence: ti.template()):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
            dudx = (u[x+1, y] - u[x, y]) / dx
            dvdy = (v[x, y+1] - v[x, y]) / dx
            divergence[x, y] = dudx + dvdy

@ti.kernel
def compute_vorticity():
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
            dudy = (u[x, y+1] - u[x, y-1]) / dx * 0.5
            dvdx = (v[x+1, y] - v[x-1, y]) / dx * 0.5
            vorticity[x, y] = dvdx - dudy

# Reset
def reset():
    density.fill(0.0)
    pressure.fill(0.0)
    apply_source()
    divergence.fill(0.0)
    vorticity.fill(0.0)
    u.fill(0.0)
    v.fill(0.0)
    f_x.fill(0.0)
    f_y.fill(0.0)

