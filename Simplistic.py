import taichi as ti
from FluidQuantity import FluidQuantity
from utils import (bilerp, 
                   copy_field,
                   swap_field,
                   euler,
                   get_value,
                   compute_divergence,
                   compute_vorticity)

@ti.data_oriented
class Fluid2d():

    def __init__(self, args):
        # Control flag
        self.wind_on = args['wind_on']

        # Discretization parameter
        self.res_x = args['res_x']             # Width
        self.res_y = args['res_y']             # Height
        self.dx = args['dx']                   # Square size
        self.dt = args['dt']                   # Time discretization
        self.acc = args['accuracy']            # Poisson equation accuracy
        self.max_iters = args['poisson_iters']   # For solving the Poisson equation
        self.t_curr = 0
        self.n_steps = 0
        self.MAC_on = True

        # Quantities
        # Grid, offset=(0.5, 0.5)
        self.density = FluidQuantity(self.res_x, self.res_y, 0.5, 0.5)
        self.pressure = FluidQuantity(self.res_x, self.res_y, 0.5, 0.5)
        self.divergence = FluidQuantity(self.res_x, self.res_y, 0.5, 0.5)
        self.vorticity = FluidQuantity(self.res_x, self.res_y, 0.5, 0.5)

        # MAC grid, offset=(0, 0.5)
        self.u = FluidQuantity(self.res_x + 1, self.res_y, 0, 0.5)
        self.f_x = FluidQuantity(self.res_x + 1, self.res_y, 0, 0.5)
        
        # MAC grid, offset=(0.5, 0)
        self.v = FluidQuantity(self.res_x, self.res_y+1, 0.5, 0)
        self.f_y = FluidQuantity(self.res_x, self.res_y+1, 0.5, 0)

        self.print_info()
        self.reset()

    def print_info(self):
        print("Plume simulator starts")
        print("Parameters:")
        print("Resolution: {}x{}".format(self.res_x, self.res_y))
        print("Grid size: {}".format(self.dx))
        print("Time step: {}".format(self.dt))
        print("Wind: {}".format(self.wind_on))
        print("Advection method: {}".format("SL"))
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

    @ti.kernel
    def copy_to(self, tmp: ti.template(), v: ti.template()):
        copy_field(tmp, v)

    @ti.kernel
    def add_buoyancy(self):
        """
        Bouyancy.
        No bouyancy at the bottom and the top.
        """
        scaling = 64.0 / self.f_y.q.shape[0]

        for i in range(self.f_y.q.shape[0]):
            for j in range(1, self.f_y.q.shape[1]-1):
                self.f_y.q[i, j] += 0.01 * (self.density.q[i, j-1] + self.density.q[i,j]) / 2 * scaling

    @ti.kernel
    def add_wind(self):
        """
        Wind force.
        Full of the grid and vary along the time
        """
        scaling = 64.0 / self.f_x.q.shape[1]

        r = self.t_curr // self.dt

        wind = 2e-2 * ti.cos(5e-2 * r) * ti.cos(3e-2 * r) * scaling

        for i, j in self.f_x:
            self.f_x.q[i,j] += wind

    @ti.kernel
    def apply_force(self):
        """
        Apply the force. 
        The second step in traditional grid method.
        """
        for x, y in self.u.q:
            self.u.q[x, y] += self.dt * self.f_x.q[x, y]

        for x, y in self.v.q:
            self.v.q[x, y] += self.dt * self.f_y.q[x, y]

    @ti.kernel
    def set_zero(self):
        """
        Velocity boundary condition
        u at x=0 and x=res_x is zero
        v at y=0 and y=res_y is zero
        """
        sx = self.u.q.shape[0]
        sy = self.u.q.shape[1]
        for y in range(sy):
            self.u.q[0, y] = 0
            self.u.q[sx-1, y] = 0

        sx = self.v.q.shape[0]
        sy = self.v.q.shape[1]
        for x in range(sx):
            self.v.q[x, 0] = 0
            self.v.q[x, sy-1] = 0

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
                    b = -self.divergence.q[x, y] / self.dt * rho
                    # Update in place
                    # if else is extremely slow in taichi

                    # numerator = dx2 * b
                    # denominator = 0
                    # if x > 0:
                    #     numerator += self.pressure[x-1, y]
                    #     denominator += 1
                    # if x < self.res_x-  1:
                    #     numerator += self.pressure[x+1, y]
                    #     denominator += 1
                    # if y > 0:
                    #     numerator += self.pressure[x, y-1]
                    #     denominator += 1
                    # if y < self.res_y - 1:
                    #     numerator += self.pressure[x, y+1]
                    #     denominator += 1
                    # self.pressure[x,y] = numerator / denominator
                    denominator = self.pressure.q[x-1, y]/self.pressure.q[x-1, y] + self.pressure.q[x+1, y]/self.pressure.q[x+1, y] + self.pressure.q[x, y-1]/self.pressure.q[x, y-1] + self.pressure.q[x, y+1]/self.pressure.q[x, y+1]
                    self.pressure.q[x,y] = (dx2 * b + self.pressure.q[x-1, y] + self.pressure.q[x+1, y] + self.pressure.q[x, y-1] + self.pressure.q[x, y+1]) / denominator

            # Compute the new residual, i.e. the sum of the squares of the individual residuals (squared L2-norm)
            residual = 0
            for y in range(0, self.res_y):
                for x in range(0, self.res_x):
                    b = -self.divergence.q[x,y] / self.dt * rho
                    cell_residual = b - (4 * self.pressure.q[x, y] - self.pressure.q[x-1, y] - self.pressure.q[x+1, y] - self.pressure.q[x, y-1] - self.pressure.q[x, y+1]) / dx2 
                    residual += cell_residual ** 2

            residual = ti.sqrt(residual)
            residual /= self.res_x * self.res_y

            it += 1
        
        # Uncomment this for debugging or monitoring
        # print(f"Poisson residual {residual}, takes {it} iterations")

    @ti.kernel
    def correct_velocity(self):
        rho = 1
        # Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
        for y in range(0, self.res_y):
            for x in range(0, self.res_x):
                self.u.q[x, y] = self.u.q[x, y] - self.dt / rho * (self.pressure.q[x, y] - self.pressure.q[x-1, y]) / self.dx
                self.v.q[x, y] = self.v.q[x, y] - self.dt / rho * (self.pressure.q[x, y] - self.pressure.q[x, y-1]) / self.dx

    # Integration each step
    def reset(self):
        # Reset all quantities and apply source
        self.density.reset()
        self.pressure.reset()
        self.divergence.reset()
        self.vorticity.reset()
        self.u.reset()
        self.v.reset()
        self.f_x.reset()
        self.f_y.reset()
        self.t_curr = 0
        self.n_steps = 0
        self.apply_init()

    def apply_init(self):
        self.apply_source(self.density.q, 0.45, 0.55, 0.10, 0.15, 1)
        self.apply_source(self.v.q, 0.45, 0.55, 0.10, 0.14, 1)
    
    def body_force(self):
        self.add_buoyancy()
        if self.wind_on:
            self.add_wind()

        self.apply_force()

    def projection(self):
        # Prepare the Poisson equation (r.h.s)
        compute_divergence(self.divergence.q, self.u.q, self.v.q, self.dx)

        # Projection step
        self.solve_poisson()
        self.correct_velocity()

        # Apply velocity boundary condition
        self.set_zero()

        compute_vorticity(self.vorticity.q, self.u.q, self.v.q, self.dx)

    def substep(self):
        self.apply_init()
        self.body_force()
        self.projection()
        # self.advect_SL(self.density, self.density_tmp, self.u, self.v)
        # self.advect_SL(self.u, self.u_tmp, self.u, self.v)
        # self.advect_SL(self.v, self.v_tmp, self.u, self.v)
        if self.MAC_on:
            self.density.advect_MC(self.u.q, self.v.q, self.dx, self.dt)
            self.u.advect_MC(self.u.q, self.v.q, self.dx, self.dt)
            self.v.advect_MC(self.u.q, self.v.q, self.dx, self.dt)
        else:
            self.density.advect_SL_RK3(self.u.q, self.v.q, self.dx, self.dt)
            self.u.advect_SL_RK3(self.u.q, self.v.q, self.dx, self.dt)
            self.v.advect_SL_RK3(self.u.q, self.v.q, self.dx, self.dt)
        
        self.f_x.reset()
        self.f_y.reset()
        self.t_curr += self.dt
        self.n_steps += 1
        
    