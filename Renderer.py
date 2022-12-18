import taichi as ti
import numpy as np
from Solid import CELL_SOLID

@ti.data_oriented
class Renderer():

    def __init__(self, res_x, res_y, dx, cell) -> None:
        self.res_x = res_x
        self.res_y = res_y
        self.dx = dx
        self.cell = ti.field(float, shape=(self.res_x, self.res_y))

        num_vertices = (res_x+1) * (res_y+1)
        num_triangles = 2*res_x*res_y
        self.V = ti.Vector.field(3, dtype=float, shape=num_vertices)
        self.F = ti.field(int, shape=num_triangles * 3)
        self.C = ti.Vector.field(3, dtype=float, shape=num_vertices)

        self.build_plane_mesh()
        self.draw_solid(cell)

    @ti.kernel
    def draw_solid(self, cell: ti.template()):
        self.cell.fill(0.)
        for y in range(self.res_y):
            for x in range(self.res_x):
                cnt, tot = 0.0, 0.0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if x+i >= 0 and x+i < self.res_x and y+j >= 0 and y+j < self.res_y:
                            if cell[x+i, y+j] > 0:
                                cnt += 1
                            tot += 1
                if cnt > 0:
                    self.cell[x, y] = cnt / tot


    @ti.kernel
    def build_plane_mesh(self):
        # Build vertices
            for i in self.V:
                self.V[i].xyz = i%(self.res_x+1) * self.dx, int(i/(self.res_x+1)) * self.dx, 0 

            # Build indices
            for y in range(self.res_y):
                for x in range(self.res_x):
                    quad_id = x + y * self.res_x
                    # First triangle of the square
                    self.F[quad_id*6 + 0] = x + y * (self.res_x + 1)
                    self.F[quad_id*6 + 1] = x + (y + 1) * (self.res_x + 1)
                    self.F[quad_id*6 + 2] = x + 1 + y * (self.res_x + 1)
                    # Second triangle of the square
                    self.F[quad_id*6 + 3] = x + 1 + (y + 1) * (self.res_x + 1)
                    self.F[quad_id*6 + 4] = x + 1 + y * (self.res_x + 1)
                    self.F[quad_id*6 + 5] = x + (y + 1) * (self.res_x + 1)

    @ti.kernel
    def get_color_raw(self, q: ti.template()):
        # Get per-vertex color using interpolation
        for y in range(self.res_y + 1):
            for x in range(self.res_x + 1):
                # Clamping
                x0 = max(x - 1, 0)
                x1 = min(x, self.res_x - 1)
                y0 = max(y - 1, 0)
                y1 = min(y, self.res_y - 1)

                c = (q[x0, y0] + q[x0, y1] + q[x1, y0] + q[x1, y1]) / 4
                t = self.cell[x1, y1]
                self.C[x + y * (self.res_x + 1)].xyz = c*(1.-t)+.1*t, c*(1.-t)+.2*t, c*(1.-t)+.3*t


        
    @ti.kernel
    def get_color_scaled(self, cmin: float, cmax: float, c1: ti.types.vector(3, float), c2: ti.types.vector(3, float)):
        for i in self.C:
            r = abs((self.C[i].x - cmin) / (cmax - cmin) * (c2[0] - c1[0]) + c1[0])
            g = abs((self.C[i].y - cmin) / (cmax - cmin) * (c2[1] - c1[1]) + c1[1])
            b = abs((self.C[i].z - cmin) / (cmax - cmin) * (c2[2] - c1[2]) + c1[2])
            self.C[i].xyz = r, g, b    

    def render(self, q: ti.template(), c1: ti.types.vector(3, float), c2: ti.types.vector(3, float)):
        self.get_color_raw(q)
        q_np = q.to_numpy()
        cmax = float(np.max(q_np))
        cmin = float(np.min(q_np))
        self.get_color_scaled(cmin, cmax, c1, c2)
