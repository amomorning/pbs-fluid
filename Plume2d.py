import taichi as ti
import numpy as np
from utils import (bilerp, 
                   copy_field,
                   compute_divergence,
                   compute_vorticity)

@ti.data_oriented
class Plume2d():

    def __init__(self, args):
        # Control flag
        self.wind_on = False
        self.MAC_on = False
        self.velocity_on = False
        self.reflection = False

        # Discretization parameter
        self.res_x = args['res_x']             # Width
        self.res_y = args['res_y']             # Height
        self.dx = args['dx']                   # Square size
        self.dt = args['dt']                   # Time discretization
        self.acc = args['accuracy']            # Poisson equation accuracy
        self.max_iters = args['poisson_iters']   # For solving the Poisson equation
        self.t_curr = 0
        self.n_steps = 0

        # Quantities
        # Grid, offset=(0.5, 0.5)
        self.density = ti.field(float, shape=(self.res_x, self.res_y))
        self.density_tmp = ti.field(float, shape=(self.res_x, self.res_y))
        self.density_forward = ti.field(float, shape=(self.res_x, self.res_y)) # For mc-advection
        self.density_backward = ti.field(float, shape=(self.res_x, self.res_y)) # For mc-advection
        self.pressure = ti.field(float, shape=(self.res_x, self.res_y))
        self.divergence = ti.field(float, shape=(self.res_x, self.res_y))
        self.vorticity = ti.field(float, shape=(self.res_x, self.res_y))

        # MAC grid, offset=(0, 0.5)
        self.u = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.u_tmp = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.u_forward = ti.field(float, shape=(self.res_x+1, self.res_y)) # For mc-advection
        self.u_backward = ti.field(float, shape=(self.res_x+1, self.res_y)) # For mc-advection
        self.f_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        

        # MAC grid, offset=(0.5, 0)
        self.v = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.v_tmp = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.v_forward = ti.field(float, shape=(self.res_x, self.res_y+1)) # For mc-advection
        self.v_backward = ti.field(float, shape=(self.res_x, self.res_y+1)) # For mc-advection
        self.f_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        # Indicate if solid, 0 if solid, 1 if fluid
        # self.solid = ti.field(float, shape=(self.res_x, self.res_y))

        # self.init_solid()
        self.print_info()
        self.reset()

    def print_info(self):
        print("Plume simulator starts")
        print("Parameters:")
        print("Resolution: {}x{}".format(self.res_x, self.res_y))
        print("Grid size: {}".format(self.dx))
        print("Time step: {}".format(self.dt))
        print("Wind: {}".format(self.wind_on))
        print("Advection method: {}".format("MacCormack" if self.MAC_on else "SL"))
        print("\n\n")

    @ti.kernel
    def apply_source(self, q: ti.template(), xmin: float, xmax: float, ymin: float, ymax: float, v: float):
        """
        rect: [x0, x1, y0, y1], between 0 and 1
        q: quantity to be applied source on
        v: value to apply
        """
        ixmin = int(xmin * q.shape[0])
        ixmax = int(xmax * q.shape[0])
        iymin = int(ymin * q.shape[1])
        iymax = int(ymax * q.shape[1])
        for x in range(ixmin, ixmax):
            for y in range(iymin, iymax):
                q[x, y] = v

    @ti.func
    def get_offset(self, q: ti.template()):
        ox = 0.5
        oy = 0.5
        if q.shape[0] == self.res_x + 1:
            ox = 0
        if q.shape[1] == self.res_y + 1:
            oy = 0
        
        return ox, oy

    # Find the value of a field at arbitrary point
    @ti.func
    def get_value(self, q: ti.template(), x: float, y: float) -> float:
        ox, oy = self.get_offset(q)

        sx = q.shape[0]
        sy = q.shape[1]

        # Clmap and project to bot-left corner
        fx = min(max(x-ox, 0.0), sx - 1.001)
        fy = min(max(y-oy, 0.0), sy - 1.001)
        ix = int(fx)
        iy = int(fy)

        x_weight = fx - ix
        y_weight = fy - iy

        return bilerp(x_weight, y_weight, q[ix, iy], q[ix+1, iy], q[ix, iy+1], q[ix+1, iy+1])

    # @ti.kernel
    # def init_solid(self):
    #     self.solid.fill(1)

    #     ixmin = int(0.45 * self.res_x)
    #     ixmax = int(0.55 * self.res_x)
    #     iymin = int(0.45 * self.res_y)
    #     iymax = int(0.55 * self.res_y)

    #     for x in range(ixmin, ixmax):
    #         for y in range(iymin, iymax):
    #             self.solid[x, y] = 0

    @ti.kernel
    def copy_to(self, tmp: ti.template(), v: ti.template()):
        copy_field(tmp, v)

    @ti.kernel
    def advect_SL(self, q: ti.template(), q_tmp: ti.template() ,u: ti.template(), v: ti.template()):
        ox, oy = self.get_offset(q)
        for iy in range(q.shape[1]):
            for ix in range(q.shape[0]):
                # Current position
                x = ix + ox
                y = iy + oy

                # Last position
                x_last = x - self.get_value(u, x, y) / self.dx * self.dt
                y_last = y - self.get_value(v, x, y) / self.dx * self.dt

                q_tmp[ix, iy] = self.get_value(q, x_last, y_last)

        copy_field(q_tmp, q)

    @ti.kernel
    def MC_correct(self, q: ti.template(), q_tmp: ti.template(), q_forward: ti.template(), q_backward: ti.template()):
        qmin = 1e-10
        qmax = 1e10
        for x, y in q:
            q_tmp[x, y] = q_forward[x, y] - 0.5 * (q_backward[x, y] - q[x, y])
            # Clamping will make the smoke not symmetric
            # if q_tmp[x, y] < qmin or q_tmp[x, y] > qmax:
            #     q_tmp[x, y] = q_forward[x, y]
        
        copy_field(q_tmp, q)
    
    def advect_MC(self, q: ti.template(), q_tmp: ti.template(), q_forward: ti.template(), q_backward: ti.template(), u: ti.template(), v: ti.template()):
        self.copy_to(q, q_forward)
        self.copy_to(q, q_backward)
        self.advect_SL(q_forward, q_tmp, u, v)
        self.dt *= -1
        self.advect_SL(q_backward, q_tmp, u, v)
        self.dt *= -1

        self.MC_correct(q, q_tmp, q_forward, q_backward)

    @ti.kernel
    def add_buoyancy(self):
        """
        Bouyancy.
        No bouyancy at the bottom and the top.
        """
        scaling = 64.0 / self.f_y.shape[0]

        for i in range(self.f_y.shape[0]):
            for j in range(1, self.f_y.shape[1]-1):
                self.f_y[i, j] += 0.01 * (self.density[i, j-1] + self.density[i,j]) / 2 * scaling

    @ti.kernel
    def add_wind(self):
        """
        Wind force.
        Full of the grid and vary along the time
        """
        scaling = 64.0 / self.f_x.shape[1]

        r = self.t_curr // self.dt

        f = 2e-2 * ti.cos(5e-2 * r) * ti.cos(3e-2 * r) * scaling

        for i, j in self.f_x:
            self.f_x[i,j] += f

    @ti.kernel
    def apply_force(self):
        """
        Apply the force. 
        The second step in traditional grid method.
        """
        for x, y in self.u:
            self.u[x, y] += self.dt * self.f_x[x, y]

        for x, y in self.v:
            self.v[x, y] += self.dt * self.f_y[x, y]

    @ti.kernel
    def set_zero(self):
        """
        Velocity boundary condition
        u at x=0 and x=res_x is zero
        v at y=0 and y=res_y is zero
        """
        sx = self.u.shape[0]
        sy = self.u.shape[1]
        for y in range(sy):
            self.u[0, y] = 0
            self.u[sx-1, y] = 0

        sx = self.v.shape[0]
        sy = self.v.shape[1]
        for x in range(sx):
            self.v[x, 0] = 0
            self.v[x, sy-1] = 0

    # @ti.func
    # def set_pressure(self, x: int, y: int, rhs: float):
    #     """
    #     Set the pressure with Neumann condition
    #     """
    #     numerator = rhs + self.pressure[x-1, y] * self.solid[x-1, y] + self.pressure[x+1, y] * self.solid[x+1, y] + self.pressure[x, y-1] * self.solid[x, y-1]+ self.pressure[x, y+1] * self.solid[x, y+1]
    #     denominator = self.solid[x-1, y] + self.solid[x+1, y] + self.solid[x, y-1] + self.solid[x, y+1]
    #     if denominator == 0:
    #         self.pressure[x, y] = 0
    #     else:
    #         self.pressure[x, y] = numerator / denominator

    @ti.kernel
    def solve_poisson(self):
        """
        Solve the Poisson equation for pressure using Gauss-Siedel method.
        """
        dx2 = self.dx * self.dx
        residual = self.acc + 1
        rho = 1 
        it = 0

        while residual > self.acc and it < self.max_iters:
            for y in range(0, self.res_y):
                for x in range(0, self.res_x):
                    b = -self.divergence[x, y] / self.dt * rho
                    # Update in place
                    # self.set_pressure(x, y, dx2 * b)
                    self.pressure[x,y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x+1, y] + self.pressure[x, y-1] + self.pressure[x, y+1]) / 4

            # Compute the new residual, i.e. the sum of the squares of the individual residuals (squared L2-norm)
            residual = 0
            for y in range(0, self.res_y):
                for x in range(0, self.res_x):
                    b = -self.divergence[x,y] / self.dt * rho
                    cell_residual = b - (4 * self.pressure[x, y] - self.pressure[x-1, y] - self.pressure[x+1, y] - self.pressure[x, y-1] - self.pressure[x, y+1]) / dx2 
                    residual += cell_residual ** 2

            residual = ti.sqrt(residual)
            residual /= self.res_x * self.res_y

            it += 1
        # print(f"Poisson residual {residual}, takes {it} iterations")

    @ti.kernel
    def correct_velocity(self):
        rho = 1
        # Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
        for y in range(0, self.res_y):
            for x in range(0, self.res_x):
                self.u[x, y] = self.u[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x-1, y]) / self.dx
                self.v[x, y] = self.v[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x, y-1]) / self.dx

    # Integration each step
    def reset(self):
        # Reset all quantities and apply source
        self.density.fill(0)
        self.density_tmp.fill(0)
        self.pressure.fill(0)
        self.divergence.fill(0)
        self.vorticity.fill(0)
        self.u.fill(0)
        self.u_tmp.fill(0)
        self.v.fill(0)
        self.v_tmp.fill(0)
        self.f_x.fill(0)
        self.f_y.fill(0)
        self.t_curr = 0
        self.n_steps = 0
        self.apply_init()

    @ti.kernel
    def reflect(self):
        for x, y in self.u_tmp:
            self.u_tmp[x, y] = 2 * self.u[x, y] - self.u_tmp[x, y]
        
        for x, y in self.v_tmp:
            self.v_tmp[x, y] = 2 * self.v[x, y] - self.v_tmp[x ,y]

    def apply_init(self):
        self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
        self.apply_source(self.v, 0.45, 0.55, 0.10, 0.14, 1)
    
    def body_force(self):
        self.add_buoyancy()
        if self.wind_on:
            self.add_wind()

        self.apply_force()

    def projection(self):
        # Prepare the Poisson equation (r.h.s)
        compute_divergence(self.divergence, self.u, self.v, self.dx)

        # Projection step
        self.solve_poisson()
        self.correct_velocity()

        # Apply velocity boundary condition
        self.set_zero()

        compute_vorticity(self.vorticity, self.u, self.v, self.dx)

    def substep(self):
        if self.reflection:
            self.apply_init()
            self.projection()
            self.reflect()
            self.advect_SL(self.u_tmp, self.u_tmp, self.u, self.v)
            self.advect_SL(self.v_tmp, self.v_tmp, self.u, self.v)
            self.advect_SL(self.density, self.density_tmp, self.u, self.v)
            self.copy_to(self.u_tmp, self.u)
            self.copy_to(self.v_tmp, self.v)
            self.body_force()
            self.projection()
            self.advect_SL(self.density, self.density_tmp, self.u, self.v)
            self.advect_SL(self.u, self.u_tmp, self.u, self.v)
            self.advect_SL(self.v, self.v_tmp, self.u, self.v)
            self.body_force()
            self.f_x.fill(0)
            self.f_y.fill(0)
            self.t_curr += 2*self.dt
            self.n_steps += 1
        else:
            self.apply_init()
            self.body_force()
            self.projection()
            if self.MAC_on:
                self.advect_MC(self.density, self.density_tmp, self.density_forward, self.density_forward, self.u, self.v)
                self.advect_MC(self.u, self.u_tmp, self.u_forward, self.u_backward, self.u, self.v)
                self.advect_MC(self.v, self.v_tmp, self.v_forward, self.v_backward, self.u, self.v)
            else:
                self.advect_SL(self.density, self.density_tmp, self.u, self.v)
                self.advect_SL(self.u, self.u_tmp, self.u, self.v)
                self.advect_SL(self.v, self.v_tmp, self.u, self.v)
            self.f_x.fill(0)
            self.f_y.fill(0)
            self.t_curr += self.dt
            self.n_steps += 1
        
    