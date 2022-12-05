from __future__ import annotations
import taichi as ti

@ti.dataclass
class FluidQuantityData():
    src: ti.field()
    tmp: ti.field()


@ti.data_oriented
class FluidQuantity():

    def __init__(self, res_x, res_y, ox, oy, dx) -> None:
        # Grid information
        self.res_x = res_x
        self.res_y = res_y
        self.ox = ox
        self.oy = oy
        self.dx = dx

        # Quantity storage
        # Some quantities can not be updated in place
        self.quantity = FluidQuantityData(self.res_x, self.res_y)

    @ti.func
    def at(self, x, y):
        # Clamp to 
        i = min(max(x - self.ox, 0), self.res_x)
        j = min(max(y - self.oy, 0), self.res_y)

    @ti.kernel
    def advect_SL(self, u: FluidQuantity, v: FluidQuantity):
        pass

    @ti.kernel
    def apply_source(self, rect: list, v: float):
        pass
