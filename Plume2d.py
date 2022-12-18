import taichi as ti
import numpy as np
from MICPCGSolver import MICPCGSolver
from utils import (bilerp, 
                   cerp,
                   copy_field,
                   compute_divergence,
                   compute_vorticity,
                   euler,
                   rk3)

@ti.data_oriented
class Plume2d():

    def __init__(self, args):
        # Control flag
        self.wind_on = False
        self.MAC_on = True
        self.velocity_on = False
        self.reflection = args['reflection']
        self.advection = args['advection']               # SL, MAC, FLIP
        self.interpolation = args['interpolation']       # bilerp, cerp
        self.integration = args['integration']           # euler, rk3
        self.solver = args['solver']                     # GS, CG, MIC

        # Discretization parameter
        self.res_x = args['res_x']             # Width
        self.res_y = args['res_y']             # Height
        self.dx = args['dx']                   # Square size
        self.dt = args['dt']                   # Time discretization
        self.acc = args['accuracy']            # Poisson equation accuracy
        self.max_iters = args['poisson_iters'] # For solving the Poisson equation
        self.t_curr = 0                        # Current time
        self.n_steps = 0                       # Current step

        self.preconditioning = True

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

        # For reflection
        self.u_half = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.v_half = ti.field(float, shape=(self.res_x, self.res_y+1))

        # Indicate if solid, 0 if solid, 1 if fluid
        # self.solid = ti.field(float, shape=(self.res_x, self.res_y))

        # settings for Conjugate Gradient method for solving poisson equation
        self.r = ti.field(float, shape=(self.res_x, self.res_y))
        self.d = ti.field(float, shape=(self.res_x, self.res_y))
        self.x = ti.field(float, shape=(self.res_x, self.res_y))
        self.q = ti.field(float, shape=(self.res_x, self.res_y))
        self.Ap = ti.field(float, shape=(self.res_x, self.res_y))

        #settings for MICPCG for solving poisson equation
        self.p_solver = None
        if(self.preconditioning):
            celltype  = ti.field(int, shape=(self.res_x, self.res_y))
            celltype.fill(1)
            self.p_solver = MICPCGSolver(self.res_x, self.res_y, self.u, self.v, cell_type=celltype, MIC_blending=0.0)

        # self.init_solid()
        self.print_info()
        self.reset()

        # Init by scheme
        if self.reflection:
            self.dt /= 2
        
        self.get_value = self.get_value_bilerp
        if self.interpolation == "cerp":
            self.get_value = self.get_value_cerp

        self.advect_SL = self.advect_SL_euler
        if self.integration == "rk3":
            self.advect_SL = self.advect_SL_rk3

        self.solve_poisson = self.solve_poisson_GS
        if self.solver == "CG":
            self.solve_poisson = self.solve_poisson_CG
        elif self.solver == "MIC":
            self.solve_poisson = self.solve_poisson_MIC

    def print_info(self):
        print("Plume simulator starts")
        print("Parameters:")
        print("Resolution: {}x{}".format(self.res_x, self.res_y))
        print("Grid size: {}".format(self.dx))
        print("Time step: {}".format(self.dt))
        print("Wind: {}".format(self.wind_on))
        print("Advection scheme: {}".format(self.advection))
        print("Interpolation scheme: {}".format(self.interpolation))
        print("Integration scheme: {}".format(self.integration))
        print("Solver: {}".format(self.solver))
        print("Reflection: {}".format(self.reflection))
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
    def get_value_bilerp(self, q: ti.template(), x: float, y: float) -> float:
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
    
    @ti.func
    def get_value_cerp(self,q: ti.template(), x: float, y: float):
        # Clmap and project to bot-left corner
        ox, oy = self.get_offset(q)
        sx = q.shape[0]
        sy = q.shape[1]

        fx = min(max(x - ox, 0.0), sx - 1.001)
        fy = min(max(y - oy, 0.0), sy - 1.001)
        ix = int(fx)
        iy = int(fy)

        x_weight = fx - ix
        y_weight = fy - iy
        
        #int index for calculating cerp
        x0 = max(ix - 1, 0)
        x1 = ix
        x2 = ix + 1
        x3 = min(ix + 2, sx - 1)

        y0 = max(iy - 1, 0)
        y1 = iy
        y2 = iy + 1
        y3 = min(iy + 2, sy - 1)

        q0 = cerp(q[x0,y0], q[x1,y0], q[x2,y0],q[x3,y0], x_weight)
        q1 = cerp(q[x0,y1], q[x1,y1], q[x2,y1],q[x3,y1], x_weight)
        q2 = cerp(q[x0,y2], q[x1,y2], q[x2,y2],q[x3,y2], x_weight)
        q3 = cerp(q[x0,y3], q[x1,y3], q[x2,y3],q[x3,y3], x_weight)

        return cerp(q0,q1,q2,q3,y_weight)

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
    def advect_SL_euler(self, q: ti.template(), q_tmp: ti.template() ,u: ti.template(), v: ti.template()):
        ox, oy = self.get_offset(q)
        for iy in range(q.shape[1]):
            for ix in range(q.shape[0]):
                # Current position
                x = ix + ox
                y = iy + oy

                # Last position
                x_last = euler(x, self.get_value(u, x, y) / self.dx, -self.dt)
                y_last = euler(y, self.get_value(v, x, y) / self.dx, -self.dt)

                q_tmp[ix, iy] = self.get_value(q, x_last, y_last)

        copy_field(q_tmp, q)

    
    
    
    @ti.kernel
    def advect_SL_rk3(self,q: ti.template(), q_tmp: ti.template() , u: ti.template(), v: ti.template()):
        ox, oy = self.get_offset(q)
        for iy in range(q.shape[1]):
            for ix in range(q.shape[0]):
                x = ix + ox
                y = iy + oy

                firstU = self.get_value(u, x, y) / self.dx
                firstV = self.get_value(v, x, y) / self.dx

                midX = x - 0.5 * self.dt * firstU
                midY = y - 0.5 * self.dt * firstV
                
                midU = self.get_value(u, midX, midY) / self.dx
                midV = self.get_value(v, midX, midY) / self.dx

                lastX = x - 0.75 * self.dt * midU
                lastY = y - 0.75 * self.dt * midV

                lastU = self.get_value(u, lastX, lastY) / self.dx
                lastV = self.get_value(v, lastX, lastY) / self.dx

                x_last = x - self.dt * ((2.0/9.0) * firstU + (1.0 / 3.0) * midU + (4.0 / 9.0) * lastU)
                y_last = y - self.dt * ((2.0/9.0) * firstV + (1.0 / 3.0) * midV + (4.0 / 9.0) * lastV)

                x_last = rk3(x, firstU, midU, lastU, -self.dt)
                y_last = rk3(y, firstV, midV, lastV, -self.dt)

                q_tmp[ix,iy] = self.get_value(q,x_last,y_last) #self.at(x_last,y_last) 

        copy_field(q_tmp, q)


    @ti.kernel
    def MC_correct(self, q: ti.template(), q_tmp: ti.template(), q_forward: ti.template(), q_backward: ti.template()):
        qmin = 1e-10
        qmax = 1e10
        for x, y in q:
            q_tmp[x, y] = q_forward[x, y] - 0.5 * (q_backward[x, y] - q[x, y])
            # Clamping will make the smoke not symmetric
            # if q_tmp[x, y] < qmin:
            #     q_tmp[x, y] = qmin
            # if q_tmp[x, y] > qmax:
            #     q_tmp[x, y] = qmax
        
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
    def solve_poisson_GS(self):
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
        print(f"Poisson residual {residual}, takes {it} iterations")

    
    @ti.kernel
    def solve_poisson_CG(self): # trivial jacobian preconditioner
        dx2 = self.dx * self.dx
        #residual = self.acc + 1
        rho = 1 
        self.x.fill(0)
        self.q.fill(0)
        self.r.fill(0)
        self.d.fill(0)

        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                r = - 1.0 *  self.divergence[i, j] / self.dt
                # r =  b(rhs) - Ax
                r -= (4.0 * self.x[i,j]  - self.x[i-1, j] - self.x[i+1, j] - self.x[i, j-1] - self.x[i, j+1]) / dx2

                self.r[i,j] = r
                self.d[i,j] = r / 4.0 # self.d = z
                self.q[i,j] = self.d[i,j]

        delta_new = 0.0
        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                delta_new += self.r[i,j] * self.r[i,j]
        delta_0 =  delta_new

        it = 0
        beta = 0.0
        delta_old = 1.0
        residual = 1.0
        alpha = 0.0

        self.Ap.fill(0)

        while it < self.max_iters and residual > 1e-4: #self.acc
            # version2, add jacobian preconditioner
            delta_new = 0.0
            
            # rz_old = r_T * z
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    delta_new += self.r[i,j] * self.d[i,j]
            print(f"delta of res {delta_new}")
            # beta = delta_new/delta_old          

            #  Ad = A.dot(q)
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    self.Ap[i,j] = 4.0 * self.q[i,j]  - self.q[i-1, j] - self.q[i+1, j] - self.q[i, j-1] - self.q[i, j+1]

            denom_a = 0.0
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    denom_a += self.q[i,j] * self.Ap[i,j]

            # rz_old / np.dot(np.transpose(d),Ad)
            alpha = delta_new/denom_a

            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    self.x[i,j] += alpha * self.q[i,j]


            
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    self.r[i,j] -= alpha * self.Ap[i,j]

            # z = Minv * r
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    self.d[i,j] = self.r[i,j] / 4.0


            delta_old = delta_new
            delta_new = 0.0
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    delta_new += self.r[i,j] * self.d[i,j]

            beta = delta_new/delta_old

            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    self.q[i,j] = self.d[i,j] + beta * self.q[i,j]

            residual = 0
            for y in range(0, self.res_y):
                for x in range(0, self.res_x):
                    b = -dx2 * self.divergence[x,y] / self.dt * rho
                    cell_residual = b - (4 * self.x[x, y] - self.x[x-1, y] - self.x[x+1, y] - self.x[x, y-1] - self.x[x, y+1])  
                    residual += cell_residual ** 2

            residual = ti.sqrt(residual)
            residual /= self.res_x * self.res_y

            it += 1
            print(f"Poisson CG residual {residual}, takes {it} iterations")
        copy_field(self.x, self.pressure)

    def solve_poisson_MIC(self):
        scale_A = self.dt / (self.dx * self.dx)
        scale_b = 1 / self.dx
        # if(self.preconditioning):
        self.p_solver.system_init(scale_A, scale_b)
        self.p_solver.solve(100)
        self.copy_to(self.p_solver.p, self.pressure)

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
        self.u_forward.fill(0)
        self.u_backward.fill(0)
        self.v.fill(0)
        self.v_tmp.fill(0)
        self.v_forward.fill(0)
        self.v_backward.fill(0)
        self.f_x.fill(0)
        self.f_y.fill(0)
        self.t_curr = 0
        self.n_steps = 0
        self.apply_init()

    @ti.kernel
    def reflect(self):
        for x, y in self.u_half:
            self.u_half[x, y] = 2 * self.u[x, y] - self.u_tmp[x, y]
        
        for x, y in self.v_half:
            self.v_half[x, y] = 2 * self.v[x, y] - self.v_tmp[x ,y]

    def apply_init(self):
        # self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
        # self.apply_source(self.v, 0.45, 0.55, 0.10, 0.14, 1)
        self.apply_source(self.density, 0.45, 0.55, 0.10, 0.15, 1)
        self.apply_source(self.v, 0.45, 0.55, 0.10, 0.14, 2)

    def body_force(self):
        self.add_buoyancy()
        if self.wind_on:
            self.add_wind()

        self.apply_force()
        self.set_zero()

    def advect(self):
        if self.advection == "MAC":
            self.advect_MC(self.density, self.density_tmp, self.density_forward, self.density_forward, self.u, self.v)
            self.advect_MC(self.u, self.u_tmp, self.u_forward, self.u_backward, self.u, self.v)
            self.advect_MC(self.v, self.v_tmp, self.v_forward, self.v_backward, self.u, self.v)
        elif self.advection == "FLIP":
            pass
        else:
            self.advect_SL(self.density, self.density_tmp, self.u, self.v)
            self.advect_SL(self.u, self.u_tmp, self.u, self.v)
            self.advect_SL(self.v, self.v_tmp, self.u, self.v)
    
    def advect_2(self):
        if self.advection == "MAC":
            self.advect_MC(self.density, self.density_tmp, self.density_forward, self.density_forward, self.u, self.v)
            self.advect_MC(self.u_half, self.u_tmp, self.u_forward, self.u_backward, self.u, self.v)
            self.advect_MC(self.v_half, self.v_tmp, self.v_forward, self.v_backward, self.u, self.v)
            self.copy_to(self.u_half, self.u)
            self.copy_to(self.v_half, self.v)
        elif self.advection == "FLIP":
            pass
        else:
            self.advect_SL(self.density, self.density_tmp, self.u, self.v)
            self.advect_SL(self.u_tmp, self.u_tmp, self.u, self.v)
            self.advect_SL(self.v_tmp, self.v_tmp, self.u, self.v)
            self.copy_to(self.u_tmp, self.u)
            self.copy_to(self.v_tmp, self.v)

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
            self.advect_2()
            self.body_force()
            self.projection()
            self.advect()
            self.body_force()
            self.f_x.fill(0)
            self.f_y.fill(0)
            self.t_curr += 2*self.dt
            self.n_steps += 1
        else:
            self.apply_init()
            self.body_force()
            self.projection()
            self.advect()
            self.f_x.fill(0)
            self.f_y.fill(0)
            self.t_curr += self.dt
            self.n_steps += 1
        
    