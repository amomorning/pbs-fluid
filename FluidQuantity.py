import taichi as ti
from utils import (
    euler,
    swap_field,
    copy_field,
    bilerp,
    get_value,
    cerp
)

@ti.data_oriented
class FluidQuantity():

    def __init__(self, res_x, res_y, ox, oy) -> None:
        self.res_x = res_x
        self.res_y = res_y
        self.ox = ox
        self.oy = oy
        self.q = ti.field(float, shape=(res_x, res_y)) # q for quantity
        self.q_tmp = ti.field(float, shape=(res_x, res_y)) # buffer for the quantity
        self.q_forward = ti.field(float, shape=(res_x, res_y)) # buffer for MAC advection
        self.q_backward = ti.field(float, shape=(res_x, res_y)) # buffer for MAC advection

    @ti.kernel
    def reset(self):
        self.q.fill(0)
        self.q_tmp.fill(0)

    # Get the value of the field at arbitrary point
    @ti.func
    def at(self, x, y):
        # Clmap and project to bot-left corner
        fx = min(max(x - self.ox, 0.0), self.res_x - 1.001)
        fy = min(max(y - self.oy, 0.0), self.res_y - 1.001)
        ix = int(fx)
        iy = int(fy)

        x_weight = fx - ix
        y_weight = fy - iy

        return bilerp(x_weight, y_weight, self.q[ix, iy], self.q[ix+1, iy], self.q[ix, iy+1], self.q[ix+1, iy+1])

    # Get field value at arbitrary point using cubic interpolation
    @ti.func
    def at_cerp(self, x, y):
        # Clmap and project to bot-left corner
        fx = min(max(x - self.ox, 0.0), self.res_x - 1.001)
        fy = min(max(y - self.oy, 0.0), self.res_y - 1.001)
        ix = int(fx)
        iy = int(fy)

        x_weight = fx - ix
        y_weight = fy - iy
        
        #int index for calculating cerp
        x0 = max(ix - 1, 0)
        x1 = ix
        x2 = ix + 1
        x3 = min(ix + 2, self.res_x - 1)

        y0 = max(iy - 1, 0)
        y1 = iy
        y2 = iy + 1
        y3 = min(iy + 2, self.res_y - 1)

        q0 = cerp(self.q[x0,y0], self.q[x1,y0], self.q[x2,y0],self.q[x3,y0], x_weight)
        q1 = cerp(self.q[x0,y1], self.q[x1,y1], self.q[x2,y1],self.q[x3,y1], x_weight)
        q2 = cerp(self.q[x0,y2], self.q[x1,y2], self.q[x2,y2],self.q[x3,y2], x_weight)
        q3 = cerp(self.q[x0,y3], self.q[x1,y3], self.q[x2,y3],self.q[x3,y3], x_weight)

        return cerp(q0,q1,q2,q3,y_weight)


    @ti.kernel
    def flip(self):
        swap_field(self.q, self.q_tmp)

    @ti.kernel
    def advect_SL(self, u: ti.template(), v: ti.template(), dx: float, dt: float):
        for iy in range(self.q.shape[1]):
            for ix in range(self.q.shape[0]):
                # Current position
                x = ix + self.ox
                y = iy + self.oy

                # Last position, in grid units
                x_last = euler(x, get_value(u, x, y, 0, 0.5) / dx, -dt)
                y_last = euler(y, get_value(v, x, y, 0.5, 0) / dx, -dt)
                # x_last = x - self.get_value(x, y) / dx * dt
                # y_last = y - self.get_value(x, y) / dx * dt

                #at.() can be replaced by higher order lerp method: cerp
                self.q_tmp[ix, iy] = self.at(x_last, y_last)
        
        copy_field(self.q_tmp, self.q)

    @ti.kernel
    def advect_SL_MAC(self, q: ti.template(), q_tmp: ti.template() ,u: ti.template(), v: ti.template(),dx: float, dt: float):
        for iy in range(q.shape[1]):
            for ix in range(q.shape[0]):
                # Current position
                x = ix + self.ox
                y = iy + self.oy

                # Last position, in grid units
                x_last = euler(x, get_value(u, x, y, 0, 0.5) / dx, -dt)
                y_last = euler(y, get_value(v, x, y, 0.5, 0) / dx, -dt)
                # x_last = x - self.get_value(x, y) / dx * dt
                # y_last = y - self.get_value(x, y) / dx * dt

                #at.() can be replaced by higher order lerp method: cerp
                q_tmp[ix, iy] = self.at(x_last, y_last)
        
        copy_field(q_tmp, q)


    #high order backtrace time integration step with third order RungeKutta
    @ti.kernel
    def advect_SL_RK3(self, u: ti.template(), v: ti.template(), dx: float, dt: float):
        for iy in range(self.q.shape[1]):
            for ix in range(self.q.shape[0]):
                x = ix + self.ox
                y = iy + self.oy

                firstU = get_value(u, x, y, 0, 0.5) / dx
                firstV = get_value(v, x, y, 0.5, 0) / dx

                midX = x - 0.5 * dt * firstU
                midY = y - 0.5 * dt * firstV
                
                midU = get_value(u, midX, midY, 0, 0.5) / dx
                midV = get_value(v, midX, midY, 0.5, 0) / dx

                lastX = x - 0.75 * dt * midU
                lastY = y - 0.75 * dt * midV

                lastU = get_value(u, lastX, lastY, 0, 0.5) / dx
                lastV = get_value(v, lastX, lastY, 0.5, 0) / dx

                x_last = x - dt * ((2.0/9.0) * firstU + (1.0 / 3.0) * midU + (4.0 / 9.0) * lastU)
                y_last = y - dt * ((2.0/9.0) * firstV + (1.0 / 3.0) * midV + (4.0 / 9.0) * lastV)

                self.q_tmp[ix,iy] = self.at_cerp(x_last,y_last) #self.at(x_last,y_last) 

        copy_field(self.q_tmp, self.q)

    @ti.kernel
    def copy_to(self, tmp: ti.template(), v: ti.template()):
        copy_field(tmp, v)
    
    @ti.kernel
    def MC_correct(self):
        qmin = 1e-10
        qmax = 1e10
        for x, y in self.q:
            self.q_tmp[x, y] = self.q_forward[x, y] - 0.5 * (self.q_backward[x, y] - self.q[x, y])
        
        copy_field(self.q_tmp, self.q)
    
    def advect_MC(self, u: ti.template(), v: ti.template(), dx: float, dt: float):
        self.copy_to(self.q, self.q_forward)
        self.copy_to(self.q, self.q_backward)
        self.advect_SL_MAC(self.q_forward,self.q_tmp,u, v, dx, dt)
        dt *= -1
        self.advect_SL_MAC(self.q_backward, self.q_tmp, u, v, dx, dt)
        dt *= -1

        self.MC_correct()
        