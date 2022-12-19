import taichi as ti


FLUID = 0
AIR = 1
SOLID = 2

@ti.pyfunc
def vec2(x, y):
    return ti.Vector([x, y])

@ti.func
def lerp(a: float, b: float, x: float):
    """
    Linear intERPolate between a and b for x ranging from 0 to 1
    """
    return a * (1.0 - x) + b * x

@ti.func
def bilerp(x_weight, y_weight, x00, x10, x01, x11):
    """
    BILinear intERPolation
    """
    return lerp(lerp(x00, x10, x_weight), lerp(x01, x11, x_weight), y_weight)

@ti.func
def cerp(x0, x1, x2, x3, w):
    """
    Cubic intERPolation
    """
    w_sq = w * w
    w_cu = w_sq * w

    #for clamping
    minx = min(x0, min(x1, min(x2,x3)))
    maxx = max(x0, max(x1, max(x2,x3)))

    t = x0 * (0.0 - 0.5 * w + 1.0 * w_sq - 0.5 * w_cu)\
         + x1 * (1.0 + 0.0 * w - 2.5 * w_sq + 1.5 * w_cu)\
         + x2 * (0.0 + 0.5 * w + 2.0 * w_sq - 1.5 * w_cu)\
         + x3 * (0.0 + 0.0 * w - 0.5 * w_sq + 0.5 * w_cu)

    return min(max(t, minx), maxx)

@ti.func
def euler(y0, vel, dt):
    return y0 + vel * dt

@ti.func 
def rk3(y0, k1, k2, k3, dt):
    return y0 + dt * ((2.0/9.0) * k1+ (1.0 / 3.0) * k2+ (4.0 / 9.0) * k3)

# Compute the divergence for a velocity field 
@ti.kernel
def compute_divergence(divergence: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(0, res_y):
        for x in range(0, res_x):
            dudx = (u[x+1, y] - u[x, y]) / dx
            dvdy = (v[x, y+1] - v[x, y]) / dx
            divergence[x, y] = dudx + dvdy

# Compute the vorticity for a velocity filed
@ti.kernel
def compute_vorticity(vorticity: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(1, res_y-1):
        for x in range(1, res_x-1):
            dudy = (u[x, y+1] - u[x, y-1]) / dx * 0.5
            dvdx = (v[x+1, y] - v[x-1, y]) / dx * 0.5
            vorticity[x, y] = dvdx - dudy

@ti.func
def copy_field(from_field: ti.template(), to_field: ti.template()):
    """
    Copy the value of one ti.field to another ti.field.
    This serves as an alternative to ti.field.copy_from since it can not be called in the Taichi-kernel scope
    
    Parameters:
    ----------
    from_field: ti.field
    to_field: ti.field
    """
    assert from_field.shape == to_field.shape, "Field shapes not matched!"
    for I in ti.grouped(from_field):
        to_field[I] = from_field[I]

@ti.func
def swap_field(f1: ti.template(), f2: ti.template()):
    """
    Swap two fields
    """
    assert f1.shape == f2.shape, "Field shapes not matched!"
    for I in ti.grouped(f1):
        tmp = f1[I]
        f1[I] = f2[I]
        f2[I] = tmp
    
@ti.func
def get_value(q: ti.template(), x, y, ox, oy):
    # Clmap and project to bot-left corner
    fx = min(max(x - ox, 0.0), q.shape[0] - 1.001)
    fy = min(max(y - oy, 0.0), q.shape[1] - 1.001)
    ix = int(fx)
    iy = int(fy)

    x_weight = fx - ix
    y_weight = fy - iy

    return bilerp(x_weight, y_weight, q[ix, iy], q[ix+1, iy], q[ix, iy+1], q[ix+1, iy+1])

@ti.func
def get_offset(q: ti.template(), res_x, res_y):
    ox = 0.5
    oy = 0.5
    if q.shape[0] == res_x + 1:
        ox = 0
    if q.shape[1] == res_y + 1:
        oy = 0
    
    return ox, oy

# @ti.func
# def preconditioner(a: ti.template):


