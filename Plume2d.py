import taichi as ti
from utils import (bilerp, 
                   forward_euler_step,
                   copy_field,
                   compute_divergence,
                   field_divide,
                   compute_vorticity)
import numpy as np

ti.init(arch=ti.cuda)

@ti.data_oriented
class Plume2d():

    def __init__(self, args):
        # Control flag
        self.wind_on = False
        self.MAC_on = False
        self.velocity_on = False

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
        # Grid
        self.density = ti.field(float, shape=(self.res_x, self.res_y))
        self.pressure = ti.field(float, shape=(self.res_x, self.res_y))
        self.divergence = ti.field(float, shape=(self.res_x, self.res_y))
        self.vorticity = ti.field(float, shape=(self.res_x, self.res_y))

        # MAC grid
        self.u = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.v = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.f_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.f_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        # Temporary quantities
        self.density_tmp = ti.field(float, shape=(self.res_x, self.res_y))
        self.u_tmp = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.v_tmp = ti.field(float, shape=(self.res_x, self.res_y+1))

        

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
        

    # @ti.kernel
    # def build_mesh(self):
    #     # Build vertices
    #     for i in self.V:
    #         self.V[i].xyz = i%(self.res_x+1) * self.dx, int(i/(self.res_x+1)) * self.dx, 0 

    #     # Build indices
    #     for y, x in ti.ndrange(self.res_y, self.res_x):
    #         quad_id = x + y * self.res_x
    #         # First triangle of the square
    #         self.F[quad_id*6 + 0] = x + y * (self.res_x + 1)
    #         self.F[quad_id*6 + 1] = x + (y + 1) * (self.res_x + 1)
    #         self.F[quad_id*6 + 2] = x + 1 + y * (self.res_x + 1)
    #         # Second triangle of the square
    #         self.F[quad_id*6 + 3] = x + 1 + (y + 1) * (self.res_x + 1)
    #         self.F[quad_id*6 + 4] = x + 1 + y * (self.res_x + 1)
    #         self.F[quad_id*6 + 5] = x + (y + 1) * (self.res_x + 1)

    # @ti.kernel
    # def get_colors(self):
    #     # Get per-vertex color using interpolation
    #     self.C.fill(0)
    #     cmin = self.density[0,0]
    #     cmax = cmin

    #     for y, x in ti.ndrange(self.res_y + 1, self.res_x + 1):
    #         # Clamping
    #         x0 = max(x - 1, 0)
    #         x1 = min(x, self.res_x - 1)
    #         y0 = max(y - 1, 0)
    #         y1 = min(y, self.res_y - 1)

    #         c = (self.density[x0, y0] + self.density[x0, y1] + self.density[x1, y0] + self.density[x1, y1]) / 4
    #         self.C[x + y * (self.res_x + 1)].xyz = c, c, c
    #         if c < cmin: cmin = c
    #         if c > cmax: cmax = c

    #     grey = [0.5, 0.5, 0.5]
    #     cyan = [0.6, 0.9, 0.92]

    #     for i in self.C:
    #         r = (self.C[i].x - cmin) / (cmax - cmin) * (cyan[0] - grey[0]) + grey[0]
    #         g = (self.C[i].y - cmin) / (cmax - cmin) * (cyan[1] - grey[1]) + grey[1]
    #         b = (self.C[i].z - cmin) / (cmax - cmin) * (cyan[2] - grey[2]) + grey[2]
    #         self.C[i].xyz = r, g, b    

    # Apply source
    @ti.kernel
    def apply_source(self):
        """
        A little square in the plane emit constant smoke.
        In future should be renamed `initialize()` to apply more initial conditions
        """
        xmin = int(0.45 * self.res_x)
        xmax = int(0.55 * self.res_x)
        ymin = int(0.10 * self.res_y)
        ymax = int(0.15 * self.res_y)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                self.density[x, y] = 1

    # Advection
    # Semi Lagrangian
    @ti.kernel
    def advect_density_SL(self):
        # new values stored in density_tmp

        for y in range(1, self.density.shape[1]-1):
            for x in range(1, self.density.shape[0]-1):
                # Velocity on grid point by lerp
                last_x_velocity = 0.5 * (self.u[x, y] + self.u[x+1, y])
                last_y_velocity = 0.5 * (self.v[x, y] + self.v[x+1, y])

                # Last position of the particle (in grid coordinates, that's why divided by dx)
                last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx

                # Clamping
                if last_x < 1: last_x = 1
                if last_y < 1: last_y = 1
                if last_x > self.res_x - 2: last_x = self.res_x - 2
                if last_y > self.res_y - 2: last_y = self.res_y - 2

                # Corners for bilinear interpolation
                x_low = int(last_x)
                y_low = int(last_y)
                x_high = x_low + 1
                y_high = y_low + 1

                bot_left = self.density[x_low, y_low]
                bot_right = self.density[x_high, y_low]
                top_left = self.density[x_low, y_high]
                top_right = self.density[x_high, x_high]
                
                # Bilinear interpolation weights
                x_weight = last_x - x_low
                y_weight = last_y - y_low

                self.density_tmp[x, y] = bilerp(x_weight, y_weight, 
                                                bot_left, bot_right, top_left, top_right)

        copy_field(self.density_tmp, self.density)

    @ti.kernel
    def advect_velocity_SL(self):
        """
        Advect the velocity field. Same logic as advect_density.
        """
        # Advect u
        for y in range(1, self.u.shape[1] - 1):
            for x in range(1, self.u.shape[0] - 1):
                # Velocity at MAC grid points, v is interpolated by the surrounding 4 grid points
                last_x_velocity = self.u[x, y] 
                last_y_velocity = (self.v[x, y] + self.v[x-1, y] + self.v[x-1, y+1] + self.v[x, y+1]) / 4

                # Last position of the particle (in grid coordinates, that's why divided by dx)
                last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx

                # Make sure the coordinates are inside the boundaries
                # Being conservative, one can say that the velocities are known between 1.5 and res-2.5
                # (the MAC grid is inside the known densities, which are between 1 and res - 2)
                # Clamping
                if last_x < 1.5: last_x = 1.5
                if last_y < 1.5: last_y = 1.5
                if last_x > self.res_x - 1.5: last_x = self.res_x - 1.5
                if last_y > self.res_y - 2.5: last_y = self.res_y - 2.5

                # Corners for bilinear interpolation
                x_low = int(last_x)
                y_low = int(last_y)
                x_high = x_low + 1
                y_high = y_low + 1

                bot_left = self.u[x_low, y_low]
                bot_right = self.u[x_high, y_low]
                top_left = self.u[x_low, y_high]
                top_right = self.u[x_high, x_high]

                # Bilinear interpolation weights
                x_weight = last_x - x_low
                y_weight = last_y - y_low

                self.u_tmp[x, y] = bilerp(x_weight, y_weight, 
                                          bot_left, bot_right, top_left, top_right)

        # Advect v
        for y in range(1, self.v.shape[1] - 1):
            for x in range(1, self.v.shape[0] - 1):
                # Velocity at MAC grid points, u is interpolated by the surrounding 4 grid points
                last_x_velocity = (self.u[x, y] + self.u[x+1, y] + self.u[x+1, y-1] + self.u[x, y-1]) / 4
                last_y_velocity = self.v[x,y]

                # Last position (in grid coordinates)
                last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                # last_x = x - last_x_velocity * self.dt / self.dx
                # last_y = y - last_y_velocity * self.dt / self.dx

                # Clamping
                if last_x < 1.5: last_x = 1.5
                if last_y < 1.5: last_y = 1.5
                if last_x > self.res_x - 2.5: last_x = self.res_x - 2.5
                if last_y > self.res_y - 1.5: last_y = self.res_y - 1.5

                # Corners for bilinear interpolation
                x_low = int(last_x)
                y_low = int(last_y)
                x_high = x_low + 1
                y_high = y_low + 1

                bot_left = self.v[x_low, y_low]
                bot_right = self.v[x_high, y_low]
                top_left = self.v[x_low, y_high]
                top_right = self.v[x_high, x_high]

                # Bilinear interpolation weights
                x_weight = last_x - x_low
                y_weight = last_y - y_low

                self.v_tmp[x, y] = bilerp(x_weight, y_weight, 
                                          bot_left, bot_right, top_left, top_right)

        copy_field(self.u_tmp, self.u)
        copy_field(self.v_tmp, self.v)

    @ti.kernel
    def add_buoyancy(self):
        """
        Bouyancy.
        No bouyancy at the bottom and the top.
        """
        scaling = 64.0 / self.f_y.shape[0]

        for i in range(self.f_y.shape[0]):
            for j in range(1, self.f_y.shape[1]-1):
                self.f_y[i,j] += 0.1 * (self.density[i, j-1] + self.density[i,j]) / 2 * scaling

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
            # self.u[x, y] = forward_euler_step(y_0=self.u[x, y], slope=self.f_x[x, y], dt=self.dt)
            self.u[x, y] += self.dt * self.f_x[x, y]

        for x, y in self.v:
            # self.v[x, y] = forward_euler_step(y_0=self.v[x, y], slope=self.f_y[x, y], dt=self.dt)
            self.v[x, y] += self.dt * self.f_y[x, y]

    @ti.kernel
    def set_neumann(self):
        # ???????? Problem here, seems u, v not right placed
        """
        Velocity boundary condition

        For u 
        |[0]|  |[1]|  |[2]|       |[border-3]| |[border-2]| |[border-1]|
            \\     //                  \\                   //
             \\   //                    \\                 //
              \\ //                      \\===============//

        For v 

                        |[border-1]| \\ 
                        |[border-2]| ||
                        |[border-3]| // 

                        |[2]| \\ 
                        |[1]| ||
                        |[0]| // 
        
        """
        sx = self.u.shape[0]
        sy = self.u.shape[1]
        for y in range(sy):
            self.u[0, y] = self.u[2, y]
            self.u[sx-1, y] = self.u[sx-3, y]

        sx = self.v.shape[0]
        sy = self.v.shape[1]
        for x in range(sx):
            self.v[x, 0] = self.v[x, 2]
            self.v[x, sy-1] = self.v[x, sy-3]

    @ti.kernel
    def set_zero(self):
        # ???????? Problem here, seems u, v not right placed
        """
        Velocity boundary condition
        u at ? is zero
        v at ? is zero
        """
        sx = self.u.shape[0]
        sy = self.u.shape[1]
        for x in range(sx):
            self.u[x,0] = 0
            self.u[x, sy-1] = 0

        sx = self.v.shape[0]
        sy = self.v.shape[1]
        for y in range(sy):
            self.v[0,y] = 0
            self.v[sy-1, y] = 0

    @ti.kernel
    def copy_border(self):
        """
        Pressure boundary condition:
        Copy inner boder value to border

                        |[border-1]|
                            /\ 
                            ||
                        |[border-2]|

        |[0]| <= |[1]|                |[border-2]| => |[border-1]|

                          |[ 1 ]|
                            ||
                            \/
                          |[ 0 ]|
        """
        sx = self.pressure.shape[0]
        sy = self.pressure.shape[1]
        for y in range(sy):
            self.pressure[0,y] = self.pressure[1,y]
            self.pressure[sx-1, y] = self.pressure[sx-2, y]

        for x in range(sx):
            self.pressure[x,0] = self.pressure[x,1]
            self.pressure[x,  sy-1] = self.pressure[x, sy-2]

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
            for y in range(1, self.res_y-1):
                for x in range(1, self.res_x-1):
                    b = -self.divergence[x,y] / self.dt * rho
                    # Update in place. 
                    self.pressure[x,y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x+1, y] + self.pressure[x, y-1] + self.pressure[x, y+1]) / 4
            
            residual = 0
            for y in range(1, self.res_y-1):
                for x in range(1, self.res_x-1):
                    b = -self.divergence[x,y] / self.dt * rho
                    cell_residual = b - (4 * self.pressure[x, y] - self.pressure[x-1, y] - self.pressure[x+1, y] - self.pressure[x, y-1] - self.pressure[x, y+1]) / dx2 
                    residual += cell_residual ** 2

            residual = ti.sqrt(residual)
            residual /= (self.res_x - 2) * (self.res_y - 2)

            it += 1
        print(f"Poisson residual {residual}, takes {it} iterations")

    @ti.kernel
    def correct_velocity(self):
        rho = 1
        # ???
        # Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
        for y in range(1, self.u.shape[1]-1):
            for x in range(0, self.u.shape[0]-1):
                self.u[x, y] = self.u[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x-1, y]) / self.dx

        for y in range(1, self.v.shape[1] - 1):
            for x in range(1, self.v.shape[0] - 1):
                self.v[x, y] = self.v[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x, y-1]) / self.dx

    # Integration each step
    def reset(self):
        # Reset all quantities and apply source
        self.density.fill(0)
        self.apply_source()
        self.pressure.fill(0)
        self.divergence.fill(0)
        self.vorticity.fill(0)
        self.u.fill(0)
        self.v.fill(0)
        self.f_x.fill(0)
        self.f_y.fill(0)
        self.t_curr = 0
        self.n_steps = 0

    def advect(self):
        # Using SL to solve advection equation
        self.advect_density_SL()
        self.advect_velocity_SL()
    
    def body_force(self):
        self.add_buoyancy()
        if self.wind_on:
            self.add_wind()

        self.apply_force()

    def projection(self):
        # Velocity border condition
        self.set_neumann()
        self.set_zero()

        # Prepare the Poisson equation (r.h.s)
        compute_divergence(self.divergence, self.u, self.v, self.dx)

        # Pressure border condition
        self.copy_border()

        # Projection step
        self.solve_poisson()
        self.correct_velocity()

        compute_divergence(self.divergence, self.u, self.v, self.dx)
        compute_vorticity(self.vorticity, self.u, self.v, self.dx)

    def substep(self):
        self.apply_source()
        self.body_force()
        self.projection()
        self.advect()
        self.t_curr += self.dt
        self.n_steps += 1

    


