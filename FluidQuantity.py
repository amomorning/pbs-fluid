import taichi as ti
from utils import (
    euler,
    swap_field,
    copy_field,
    bilerp,
    get_value
)

@ti.data_oriented
class FluidQuantity():

    def __init__(self, res_x, res_y, ox, oy) -> None:
        self.res_x = res_x
        self.res_y = res_y
        self.ox = ox
        self.oy = oy
        self.q = ti.field(float, shape=(res_x, res_y)) # q for quantity
        self.q_tmp = ti.field(float, shape=(res_x, res_y)) # buffer for the quantity
    
    @ti.kernel
    def reset(self):
        self.q.fill(0)
        self.q_tmp.fill(0)

    # Get the value of the field at arbitrary point
    @ti.func
    def at(self, x, y):
        # Clmap and project to bot-left corner
        fx = min(max(x - self.ox, 0.0), self.res_x - 1.001)
        fy = min(max(y - self.oy, 0.0), self.res_y - 1.001)
        ix = int(fx)
        iy = int(fy)

        x_weight = fx - ix
        y_weight = fy - iy

        return bilerp(x_weight, y_weight, self.q[ix, iy], self.q[ix+1, iy], self.q[ix, iy+1], self.q[ix+1, iy+1])

    @ti.kernel
    def flip(self):
        swap_field(self.q, self.q_tmp)

    @ti.kernel
    def advect_SL(self, u: ti.template(), v: ti.template(), dx: float, dt: float):
        for iy in range(self.q.shape[1]):
            for ix in range(self.q.shape[0]):
                # Current position
                x = ix + self.ox
                y = iy + self.oy

                # Last position, in grid units
                x_last = euler(x, get_value(u, x, y, 0, 0.5) / dx, -dt)
                y_last = euler(y, get_value(v, x, y, 0.5, 0) / dx, -dt)
                # x_last = x - self.get_value(x, y) / dx * dt
                # y_last = y - self.get_value(x, y) / dx * dt

                self.q_tmp[ix, iy] = self.at(x_last, y_last)
        
        copy_field(self.q_tmp, self.q)