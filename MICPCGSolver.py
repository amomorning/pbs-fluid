import taichi as ti
from Solid import CELL_FLUID, CELL_SOLID, CELL_AIR


@ti.data_oriented
class MICPCGSolver:
    def __init__(self, res_x, res_y, u, v, cell_type, MIC_blending=0.0):
        self.res_x = res_x
        self.res_y = res_y
        self.u = u
        self.v = v
        self.cell_type = cell_type # int field for celltype indication, 0 for fluid, 1 for solid
        self.MIC_blending = MIC_blending

        # rhs of linear system
        self.b = ti.field(float, shape=(self.res_x, self.res_y))

        # lhs of linear system
        self.Adiag = ti.field(float, shape=(self.res_x, self.res_y))
        self.Ax = ti.field(float, shape=(self.res_x, self.res_y))
        self.Ay = ti.field(float, shape=(self.res_x, self.res_y))

        # cg var
        self.p = ti.field(float, shape=(self.res_x, self.res_y))
        self.r = ti.field(float, shape=(self.res_x, self.res_y))
        self.s = ti.field(float, shape=(self.res_x, self.res_y))
        self.As = ti.field(float, shape=(self.res_x, self.res_y))
        self.sum = 0.0
        self.alpha = 0.0
        self.beta = 0.0

        # MIC precondition
        self.precon = ti.field(float, shape=(self.res_x, self.res_y))
        self.z = ti.field(float, shape=(self.res_x, self.res_y))
        self.q = ti.field(float, shape=(self.res_x, self.res_y))

        print("Resolution of PCG: {}x{}".format(self.res_x, self.res_y))

    @ti.kernel
    def system_init_kernel(self, scale_A: float, scale_b: float):
        #define right hand side of linear system
        #scale_b for 1/dx
        #scale_A for dt/(rho * dx2)
        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                if self.cell_type[i, j] == CELL_FLUID:
                    self.b[i,j] = -1 * scale_b * (self.u[i + 1, j] - self.u[i, j] + self.v[i, j + 1] - self.v[i, j]) 
                    # compute for the div(can be replace by self.div)

        #modify right hand side of linear system to account for solid velocities
        #currently hard code solid velocities to zero

        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                if self.cell_type[i, j] == CELL_FLUID:
                    if self.cell_type[i - 1, j] == CELL_SOLID:
                        self.b[i, j] -= scale_b * (self.u[i, j] - 0)
                    if self.cell_type[i + 1, j] == CELL_SOLID:
                        self.b[i, j] += scale_b * (self.u[i + 1, j] - 0)

                    if self.cell_type[i, j - 1] == CELL_SOLID:
                        self.b[i, j] -= scale_b * (self.v[i, j] - 0)
                    if self.cell_type[i, j + 1] == CELL_SOLID:
                        self.b[i, j] += scale_b * (self.v[i, j + 1] - 0)

        # define left handside of linear system
        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                if self.cell_type[i, j] == CELL_FLUID:
                    if self.cell_type[i - 1, j] == CELL_FLUID:
                        self.Adiag[i, j] += scale_A
                    if self.cell_type[i + 1, j] == CELL_FLUID:
                        self.Adiag[i, j] += scale_A
                        self.Ax[i, j] = -scale_A
                    elif self.cell_type[i + 1, j] == CELL_AIR:
                        self.Adiag[i, j] += scale_A

                    if self.cell_type[i, j - 1] == CELL_FLUID:
                        self.Adiag[i, j] += scale_A
                    if self.cell_type[i, j + 1] == CELL_FLUID:
                        self.Adiag[i, j] += scale_A
                        self.Ay[i, j] = -scale_A
                    elif self.cell_type[i, j + 1] == CELL_AIR:
                        self.Adiag[i, j] += scale_A
        # print("successfully init PCG kernel")

    @ti.kernel
    def preconditioner_init(self):
        sigma = 0.25  # safety constant

        for _ in range(1):  # force serial
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    if self.cell_type[i, j] == CELL_FLUID:
                        e = self.Adiag[i, j] - (\
                            self.Ax[i - 1, j] * self.precon[i - 1, j])**2 - (\
                                self.Ay[i, j - 1] *\
                                self.precon[i, j - 1])**2 - self.MIC_blending * (\
                                    self.Ax[i - 1, j] * self.Ay[i - 1, j] *\
                                    self.precon[i - 1, j]**2 + self.Ay[i, j - 1] *\
                                    self.Ax[i, j - 1] * self.precon[i, j - 1]**2)

                        if e < sigma * self.Adiag[i, j]:
                            e = self.Adiag[i, j]

                        self.precon[i, j] = 1 / ti.sqrt(e)
        # print("successfully init PCG preconditioner")

    def system_init(self, scale_A, scale_b):
        self.b.fill(0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.precon.fill(0.0)

        self.system_init_kernel(scale_A, scale_b)
        self.preconditioner_init()
        # print("successfully init PCG")

    def solve(self, max_iters):
        tol = 1e-5 #set tolerance

        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r.copy_from(self.b)

        
        init_rTr = self.reduce(self.r, self.r)

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # z0 = M^-1r0
            self.q.fill(0.0)
            self.z.fill(0.0)
            self.applyPreconditioner()

            # s0 = z0
            self.s.copy_from(self.z)

            # zTr

            
            old_zTr = self.reduce(self.z, self.r)

            iteration = 0
            rTr = old_zTr

            while (iteration < max_iters and rTr > init_rTr * tol):
                # alpha = zTr / sAs
                self.compute_As()

                
                sAs = self.reduce(self.s, self.As)
                self.alpha = old_zTr / sAs

                # p = p + alpha * s
                self.update_p()

                # r = r - alpha * As
                self.update_r()

                # check for convergence
                
                rTr = self.reduce(self.r, self.r)
                # if rTr < init_rTr * tol:
                #     break

                # z = M^-1r
                self.q.fill(0.0)
                self.z.fill(0.0)
                self.applyPreconditioner()

                
                new_zTr = self.reduce(self.z, self.r)

                # beta = zTrnew / zTrold
                self.beta = new_zTr / old_zTr

                # s = z + beta * s
                self.update_s()
                old_zTr = new_zTr
                iteration += 1

                # if iteration % 100 == 0:
                #     print("iter {}, res = {}".format(iteration, rTr))

            print("Converged to {} in {} iterations".format(rTr, iteration))

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()) -> ti.f32:
        sum = 0.0
        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                if self.cell_type[i, j] == CELL_FLUID:
                    sum += p[i, j] * q[i, j]
        return sum

    @ti.kernel
    def compute_As(self):
        for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    if self.cell_type[i, j] == CELL_FLUID:
                        self.As[i, j] = self.Adiag[i, j] * self.s[i, j] + self.Ax[
                            i - 1, j] * self.s[i - 1, j] + self.Ax[i, j] * self.s[
                                i + 1, j] + self.Ay[i, j - 1] * self.s[
                                    i, j - 1] + self.Ay[i, j] * self.s[i, j + 1]

    @ti.kernel
    def update_p(self):
        for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    if self.cell_type[i, j] == CELL_FLUID:
                        self.p[i, j] = self.p[i, j] + self.alpha * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    if self.cell_type[i, j] == CELL_FLUID:
                        self.r[i, j] = self.r[i, j] - self.alpha * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i in range(0, self.res_y): 
            for j in range(0, self.res_x):
                if self.cell_type[i, j] == CELL_FLUID:
                  self.s[i, j] = self.z[i, j] + self.beta * self.s[i, j]

    @ti.kernel
    def applyPreconditioner(self):
        # # first solve Lq = r
        for _ in range(1):
            for i in range(0, self.res_y): 
                for j in range(0, self.res_x):
                    if self.cell_type[i, j] == CELL_FLUID:
                        t = self.r[i, j] - self.Ax[i - 1, j] * self.precon[
                            i - 1, j] * self.q[i - 1, j] - self.Ay[
                                i, j - 1] * self.precon[i, j - 1] * self.q[i, j - 1]

                        self.q[i, j] = t * self.precon[i, j]

        # next solve LTz = q
        for _ in range(1):
            for iy in range(0, self.res_y): 
                for ix in range(0, self.res_x):
                    i = self.res_x - 1 - ix
                    j = self.res_y - 1 - iy

                    if self.cell_type[i, j] == CELL_FLUID:
                        t = self.q[i, j] - self.Ax[i, j] * self.precon[
                            i, j] * self.z[i + 1, j] - self.Ay[i, j] * self.precon[
                                i, j] * self.z[i, j + 1]

                        self.z[i, j] = t * self.precon[i, j]
