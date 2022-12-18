import taichi as ti
import math

CELL_FLUID = 0
CELL_SOLID = 1 

def nsgn(x):
    return -1.0 if x < 0.0 else 1.0

def length(x, y):
    return ti.sqrt(x*x + y*y) 

def rotate(x, y, angle):
    return ti.cos(angle) * x + ti.sin(angle) * y,\
          -ti.sin(angle) * x + ti.cos(angle) * y

class SolidBody:
    
    def __init__(self, posX, posY, scaleX, scaleY, theta, velX, velY, velTheta):
        self._posX = posX
        self._posY = posY
        self._scaleX = scaleX
        self._scaleY = scaleY
        self._theta = theta
        self._velX = velX
        self._velY = velY
        self._velTheta = velTheta
    
    def global_to_local(self, x, y):
        x -= self._posX
        y -= self._posY
        x, y = rotate(x, y, -self._theta)
        x /= self._scaleX
        y /= self._scaleY
        return x, y
    
    def local_to_global(self, x, y):
        x *= self._scaleX
        y *= self._scaleY
        x, y = rotate(x, y, self._theta)
        x += self._posX
        y += self._posY
        return x, y
        
    def get_velocityX(self, x, y):
        return (self._posY - y) * self._velTheta + self._velX

    def get_velocityY(self, x, y):
        return (x - self._posX) * self._velTheta + self._velY

class SolidBox(SolidBody):
    def __init__(self, x, y, sx=1.0, sy=1.0, t=0.0, vx=0.0, vy=0.0, vt=0.0):
        super().__init__(x, y, sx, sy, t, vx, vy, vt)

    def distance(self, x, y):
        x -= self._posX
        y -= self._posY
        x, y = rotate(x, y, -self._theta)
        dx = ti.abs(x) - self._scaleX*0.5
        dy = ti.abs(y) - self._scaleY*0.5

        if dx >= 0.0 or dy >= 0.0:
            return length(max(dx, 0.0), max(dy, 0.0))
        else:
            return max(dx, dy)

    def closestSurfacePoint(self, x, y):
        x -= self._posX
        y -= self._pos
        x, y = rotate(x, y, -self._theta)
        dx = ti.abs(x) - self._scaleX * 0.5
        dy = ti.abs(y) - self._scaleY * 0.5

        if dx > dy:
            x = nsgn(x) * 0.5 * self._scaleX
        else:
            y = nsgn(y) * 0.5 * self._scaleY
        
        x, y = rotate(x, y, self._theta)
        x += self._posX
        y += self._posY

        return x, y
        
class SolidSphere(SolidBody):
    def __init__(self, x, y, s=1.0, t=0.0, vx=0.0, vy=0.0, vt=0.0):
        super().__init__(x, y, s, s, t, vx, vy, vt)

    def distance(self, x, y):
        return length(x-self._posX, y-self._posY) - self._scaleX*0.5
    
    def closetSurfacePoint(self, x, y):
        x -= self._posX
        y -= self._posY
        r = length(x, y)
        if r < 1e-4:
            nx, ny = 1.0, 0.0
        else:
            nx, ny = x/r, y/r
        return nx, ny
    

if __name__ == '__main__':
    box = SolidBox(0, 0, 1, 1, 0.25*math.pi)
    print(f'{box.distance(.25, .25)=}')
    print(f'{box.distance(0, 1.2)=}')

    sph = SolidSphere(0, 0)
    print(f'{sph.distance(.25, .25)=}')
