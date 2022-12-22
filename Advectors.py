import taichi as ti
import utils
import random


@ti.data_oriented
class SemiLagrangian():

    def __init__(self) -> None:
        pass

@ti.data_oriented
class MacCormack(SemiLagrangian):

    def __init__(self) -> None:
        pass


@ti.data_oriented
class FLIP():

    MaxPerCell = 12
    MinPerCell = 3
    AvgPerCell = 4

    def __init__(self, res_x, res_y, dx) -> None:
        self.res_x = res_x
        self.res_y = res_y
        self.dx = dx

        self.maxParticles = res_x * res_y * self.AvgPerCell # Maximum particles could exist
        self.particleCount = res_x * res_y * self.AvgPerCell # Current particle count

        self.weight = ti.field(float, shape=(self.res_x, self.res_y))
        self.particlePosition = ti.Vector.field(2, dtype=float ,shpae=self.maxParticles)
        
        

        
    