import taichi as ti
import utils
from utils import vec2
from Solid import CELL_FLUID
import random

@ti.data_oriented
class Particles():

    def __init__(self, npar, res_x, res_y, dx) -> None: # n particles per edge
        
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=(res_x, res_y, npar, npar))
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=(res_x, res_y, npar, npar))
        self.type = ti.Vector.field(dtype=ti.f32, shape=(res_x, res_y, npar, npar))
        self.res_x = res_x
        self.res_y = res_y
        self.dx = dx
        self.pspace_x = dx / npar
        self.pspace_y = dx / npar


        self.cp_x = ti.Vector.field(2, dtype=ti.f32, shape=(res_x, res_y, npar, npar))
        self.cp_y = ti.Vector.field(2, dtype=ti.f32, shape=(res_x, res_y, npar, npar))

    @ti.kernel
    def init_particles(self, cell_type):
        for i, j, ix, jx in self.positions:
            if cell_type[i, j] == utils.FLUID:
                self.type[i, j, ix, jx] = CELL_FLUID
            else:
                self.type[i, j, ix, jx] = 0

            px = i * self.dx + (ix + random.random()) * self.pspace_x
            py = j * self.dx + (jx + random.random()) * self.pspace_y

            self.positions[i, j, ix, jx] = vec2(px, py)
            self.velocities[i, j, ix, jx] = vec2(0.0, 0.0)
            self.cp_x[i, j, ix, jx] = vec2(0.0, 0.0)
            self.cp_y[i, j, ix, jx] = vec2(0.0, 0.0)

    @ti.kernel
    def update_particle_velocities(self, u: ti.template(), v:ti.template()):
        for p in ti.grouped(self.positions):
            if self.type[p] == CELL_FLUID:
                self.velocities[p] = utils.get_value(self.positions[p], u, v)
    