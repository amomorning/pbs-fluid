import taichi as ti

@ti.func
def lerp(a: float, b: float, x: float):
    """
    Linear interpolate between a and b for x ranging from 0 to 1
    """
    return a * (1.0 - x) + b * x

@ti.func
def bilerp(x_weight, y_weight, x00, x10, x01, x11):
    """
    Bilinear interpolation
    """
    return lerp(lerp(x00, x10, x_weight), lerp(x01, x11, x_weight), y_weight)

# Compute the divergence for a velocity field 
@ti.kernel
def compute_divergence(divergence: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
            dudx = (u[x+1, y] - u[x, y]) / dx
            dvdy = (v[x, y+1] - v[x, y]) / dx
            divergence[x, y] = dudx + dvdy

# Compute the vorticity for a velocity filed
@ti.kernel
def compute_vorticity(vorticity: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
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