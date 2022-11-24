import taichi as ti

@ti.func
def bilerp(x_weight, y_weight, bot_left, bot_right, top_left, top_right) -> float:
    """
    Calculate bilinear interpolation given weights and corners:
    
    Parameters:
    ----------
    weights: x_weight, y_weight
    corners: bot_left, bot_right, top_left, top_right

    Return
    ------
    Bilinear interpolation of the square
    """
    a1 = (1 - x_weight) * (1 - y_weight) * bot_left
    a2 = x_weight * (1 - y_weight) * bot_right
    a3 = (1 - x_weight) * y_weight * top_left
    a4 = x_weight * y_weight * top_right

    return a1+a2+a3+a4

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
    for x, y in from_field:
        to_field[x, y] = from_field[x, y]

@ti.func
def field_multiply(field: ti.template(), scalar: float):
    for I in ti.grouped(field):
        field[I] *= scalar

@ti.func
def field_divide(field: ti.template(), scalr: float):
    assert scalr != 0, "Divided by zero"
    field_multiply(field, 1/scalr)

@ti.func
def forward_euler_step(y_0: float, slope: float, dt: float) -> float:
    return y_0 + slope * dt

@ti.kernel
def compute_divergence(divergence: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
            dudx = (u[x+1, y] - u[x, y]) / dx
            dvdy = (v[x, y+1] - v[x, y]) / dx
            divergence[x, y] = dudx + dvdy

@ti.kernel
def compute_vorticity(vorticity: ti.template(), u: ti.template(), v: ti.template(), dx: float):
    res_y = u.shape[1]
    res_x = v.shape[0]
    for y in range(2, res_y-2):
        for x in range(2, res_x-2):
            dudy = (u[x, y+1] - u[x, y-1]) / dx * 0.5
            dvdx = (v[x+1, y] - v[x-1, y]) / dx * 0.5
            vorticity[x, y] = dvdx - dudy