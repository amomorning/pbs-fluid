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

# @ti.func
# def lerp(a: float, b: float, x: float):
#     """
#     Linear interpolate between a and b for x ranging from 0 to 1
#     """
#     return a * (1.0 - x) + b * x

# @ti.func
# def bilerp(x_weight, y_weight, x00, x10, x01, x11):
#     """
#     Bilinear interpolation
#     """
#     return lerp(lerp(x00, x10, x_weight), lerp(x01, x11, x_weight), y_weight)

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
    
@ti.kernel
def build_plane_mesh(V: ti.template(), F: ti.template(), res_x: int, res_y: int, dx: float):
    # Build vertices
        for i in V:
            V[i].xyz = i%(res_x+1) * dx, int(i/(res_x+1)) * dx, 0 

        # Build indices
        for y in range(res_y):
            for x in range(res_x):
                quad_id = x + y * res_x
                # First triangle of the square
                F[quad_id*6 + 0] = x + y * (res_x + 1)
                F[quad_id*6 + 1] = x + (y + 1) * (res_x + 1)
                F[quad_id*6 + 2] = x + 1 + y * (res_x + 1)
                # Second triangle of the square
                F[quad_id*6 + 3] = x + 1 + (y + 1) * (res_x + 1)
                F[quad_id*6 + 4] = x + 1 + y * (res_x + 1)
                F[quad_id*6 + 5] = x + (y + 1) * (res_x + 1)

@ti.kernel
def get_plane_colors(C: ti.template(), q: ti.template(), res_x: int, res_y: int):
    # Get per-vertex color using interpolation
    cmin: ti.f32 = 0
    cmax: ti.f32 = 1
    # cmin = q[0, 0]
    # cmax = q[0, 0]

    for y in range(res_y + 1):
        for x in range(res_x + 1):
            # Clamping
            x0 = max(x - 1, 0)
            x1 = min(x, res_x - 1)
            y0 = max(y - 1, 0)
            y1 = min(y, res_y - 1)

            c = (q[x0, y0] + q[x0, y1] + q[x1, y0] + q[x1, y1]) / 4
            C[x + y * (res_x + 1)].xyz = c, c, c
            # if c < cmin: cmin = c
            # if c > cmax: cmax = c

    color1 = [0, 0, 0]
    color2 = [1, 1, 1]

    for i in C:
        r = (C[i].x - cmin) / (cmax - cmin) * (color2[0] - color1[0]) + color1[0]
        g = (C[i].y - cmin) / (cmax - cmin) * (color2[1] - color1[1]) + color1[1]
        b = (C[i].z - cmin) / (cmax - cmin) * (color2[2] - color1[2]) + color1[2]
        C[i].xyz = r, g, b    