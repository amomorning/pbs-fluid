import taichi as ti
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
        self.pressure = ti.field(float, shape=(self.res_x, self.res_y))
        self.divergence = ti.field(float, shape=(self.res_x, self.res_y))
        self.vorticity = ti.field(float, shape=(self.res_x, self.res_y))

        # MAC grid, offset=(0, 0.5)
        self.u = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.u_half = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.u_tmp = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.f_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        

        # MAC grid, offset=(0.5, 0)
        self.v = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.v_half = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.v_tmp = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.f_y = ti.field(float, shape=(self.res_x, self.res_y+1))

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

    # Apply source
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

        copy_field(self.u, self.u_tmp)
        copy_field(self.v, self.v_tmp)

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
    def advect_velocity_tmp(self, u: ti.template(), v: ti.template()):
        for y in range(1, self.u.shape[1] - 1):
            for x in range(1, self.u.shape[0] - 1):
                # Velocity at MAC grid points, v is interpolated by the surrounding 4 grid points
                last_x_velocity = self.u[x, y] 
                last_y_velocity = (self.v[x, y] + self.v[x-1, y] + self.v[x-1, y+1] + self.v[x, y+1]) / 4

                # Last position of the particle (in grid coordinates, that's why divided by dx)
                # last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                # last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                last_x = x - last_x_velocity / self.dx * self.dt
                last_y = y - last_y_velocity / self.dx * self.dt


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

                bot_left = u[x_low, y_low]
                bot_right = u[x_high, y_low]
                top_left = u[x_low, y_high]
                top_right = u[x_high, y_high]

                # Bilinear interpolation weights
                x_weight = last_x - x_low
                y_weight = last_y - y_low

                self.u_tmp[x, y] = bilerp(x_weight, y_weight,
                                          bot_left, bot_right, top_left, top_right)

        # Advect v
        for y in range(1, self.v.shape[1] - 1):
            for x in range(1, self.v.shape[0] - 1):
                # Velocity at MAC grid points, u is interpolated by the surrounding 4 grid points
                last_x_velocity = (u[x, y] + u[x+1, y] + u[x+1, y-1] + u[x, y-1]) / 4
                last_y_velocity = v[x,y]

                # Last position (in grid coordinates)
                # last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                # last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                last_x = x - last_x_velocity * self.dt / self.dx
                last_y = y - last_y_velocity * self.dt / self.dx

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
                top_right = self.v[x_high, y_high]

                # Bilinear interpolation weights
                x_weight = last_x - x_low
                y_weight = last_y - y_low

                self.v_half[x, y] = bilerp(x_weight, y_weight,
                                          bot_left, bot_right, top_left, top_right)
        copy_field(self.u_tmp, self.u)
        copy_field(self.v_tmp, self.v)

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
                # last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                # last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                last_x = x - last_x_velocity / self.dx * self.dt
                last_y = y - last_y_velocity / self.dx * self.dt

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
                top_right = self.density[x_high, y_high]
                
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
                # last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                # last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                last_x = x - last_x_velocity / self.dx * self.dt
                last_y = y - last_y_velocity / self.dx * self.dt


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
                top_right = self.u[x_high, y_high]

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
                # last_x = forward_euler_step(y_0=x*self.dx, slope=last_x_velocity, dt=-self.dt) / self.dx
                # last_y = forward_euler_step(y_0=y*self.dx, slope=last_y_velocity, dt=-self.dt) / self.dx
                last_x = x - last_x_velocity * self.dt / self.dx
                last_y = y - last_y_velocity * self.dt / self.dx

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
                top_right = self.v[x_high, y_high]

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
                self.f_y[i, j] += 0.1 * (self.density[i, j-1] + self.density[i,j]) / 2 * scaling


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

    # @ti.kernel
    # def set_vel_boundary(self):
    #     # Corners have zero velocity
    #     for x, y in self.u:
    #         if ((x == 0 and y == 0) 
    #          or (x == 0 and y == self.u.shape[1]-1)
    #          or (x == self.u.shape[0]-1 and y == 0)
    #          or (x == self.u.shape[0]-1 and y == self.u.shape[1]-1)):
    #             self.u[x, y] = 0
    #         elif x == 0:
    #             self.u[x, y] = -self.u[x+1, y]
    #         elif x == self.u.shape[0] - 1:
    #             self.u[x, y] = -self.u[x-1, y]
        
    #     for x, y in self.v:
    #         if ((x == 0 and y == 0) 
    #          or (x == 0 and y == self.u.shape[1]-1)
    #          or (x == self.u.shape[0]-1 and y == 0)
    #          or (x == self.u.shape[0]-1 and y == self.u.shape[1]-1)):
    #             self.v[x, y] = 0
    #         elif y == 0:
    #             self.v[x, y] = -self.v[x, y+1]
    #         elif y == self.v.shape[1] - 1:
    #             self.v[x, y] = -self.v[x, y-1]



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
            for y in range(0, self.res_y):
                for x in range(0, self.res_x):
                    b = -self.divergence[x, y] / self.dt * rho
                    # Update in place. 
                    if x == 0 and y == 0:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x+1, y] + self.pressure[x, y+1]) / 2
                    elif x == 0 and y == self.res_y-1:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x+1, y] + self.pressure[x, y-1]) / 2
                    elif x == self.res_x-1 and y == 0:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x, y+1]) / 2
                    elif x == self.res_x-1 and y == self.res_y-1:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x, y-1]) / 2
                    elif x == 0: 
                        self.pressure[x, y] = (dx2 * b + self.pressure[x+1, y] + self.pressure[x, y-1] + self.pressure[x, y+1]) / 3
                    elif x == self.res_x-1:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x, y-1] + self.pressure[x, y+1]) / 3
                    elif y == 0: 
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x+1, y] + self.pressure[x, y+1]) / 3
                    elif y == self.res_y-1:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x+1, y] + self.pressure[x, y-1]) / 3
                    else:
                        self.pressure[x, y] = (dx2 * b + self.pressure[x-1, y] + self.pressure[x+1, y] + self.pressure[x, y-1] + self.pressure[x, y+1]) / 4

            # Compute the new residual, i.e. the sum of the squares of the individual residuals (squared L2-norm)
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
        # Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
        for y in range(0, self.res_x):
            for x in range(0, self.res_y):
                self.u[x, y] = self.u[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x-1, y]) / self.dx
                self.v[x, y] = self.v[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x, y-1]) / self.dx

        # for y in range(0, self.v.shape[1] - 1):
        #     for x in range(0, self.v.shape[0] - 1):
        #         self.v[x, y] = self.v[x, y] - self.dt / rho * (self.pressure[x, y] - self.pressure[x, y-1]) / self.dx

    # Integration each step
    def reset(self):
        # Reset all quantities and apply source
        self.density.fill(0)
        self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
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
        self.advect_SL(self.density, self.density_tmp, self.u, self.v)
        self.advect_SL(self.u, self.u_tmp, self.u, self.v)
        self.advect_SL(self.v, self.v_tmp, self.u, self.v)
        # self.advect_density_SL()
        # self.advect_velocity_SL()

    @ti.kernel
    def reflect(self):
        for x, y in self.u_half:
            self.u_half[x, y] = 2 * self.u[x, y] - self.u_tmp[x, y]
        
        for x, y in self.v_half:
            self.v_half[x, y] = 2 * self.v[x, y] - self.v_tmp[x ,y]
    
    def body_force(self):
        self.add_buoyancy()
        if self.wind_on:
            self.add_wind()

        self.apply_force()

    def projection(self):
        # Velocity border condition
        # self.set_neumann()
        

        # Prepare the Poisson equation (r.h.s)
        compute_divergence(self.divergence, self.u, self.v, self.dx)

        # # Pressure border condition
        # self.copy_border()

        # Projection step
        self.solve_poisson()
        self.correct_velocity()

        # Apply velocity boundary condition
        self.set_zero()

        compute_divergence(self.divergence, self.u, self.v, self.dx)
        compute_vorticity(self.vorticity, self.u, self.v, self.dx)

    def substep(self):
        self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
        self.body_force()
        self.projection()
        self.advect()
        self.f_x.fill(0)
        self.f_y.fill(0)
        self.t_curr += self.dt
        self.n_steps += 1

    def substep_reflection(self):
        self.dt /= 2
        self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
        # self.body_force()
        self.projection()
        self.reflect()
        self.advect_velocity_tmp(self.u_half, self.v_half)
        self.advect_density_SL()
        self.projection()
        self.advect()
        self.f_x.fill(0)
        self.f_y.fill(0)
        self.t_curr += 2*self.dt
        self.n_steps += 1
    