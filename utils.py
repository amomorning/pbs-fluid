import taichi as ti

@ti.func
def bilerp():
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
    pass

@ti.func
def copy_field(from_field: ti.template(), to_field: ti.template()):
    """
    Copy the value of one ti.field to another ti.field
    
    Parameters:
    ----------
    from_field: ti.field
    to_field: ti.field
    """
    assert from_field.shape == to_field.shape, "Field shapes not matched!"
    for x, y in from_field:
        to_field[x, y] = from_field[x, y]


@ti.func
def forward_euler_step(y_0: float, slope: float, dt: float):
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