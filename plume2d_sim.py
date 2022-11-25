import taichi as ti
import numpy as np
from utils import (
    build_plane_mesh,
    get_plane_colors
)
from Plume2d import Plume2d

res_x = 128
res_y = int(res_x * 1.5)
dx = 1
dt = 0.005 * ti.sqrt((res_x + res_y) * 0.5)
accuracy = 1e-5
n_iters = 1000

args = {
    'res_x': res_x,
    'res_y': res_y,
    'dx': dx,
    'dt': dt,
    'accuracy': accuracy,
    'poisson_iters': n_iters
}

plume = Plume2d(args)

# For rendering
num_vertices = (res_x+1) * (res_y+1)
num_triangles = 2*res_x*res_y
V = ti.Vector.field(3, dtype=float, shape=num_vertices)
F = ti.field(int, shape=num_triangles * 3)
C = ti.Vector.field(3, dtype=float, shape=num_vertices)

build_plane_mesh(V, F, res_x, res_y, dx)

window = ti.ui.Window("Plume 2d", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    if plume.t_curr > 200:
        # Reset
        plume.reset()

    plume.substep()
    get_plane_colors(C, plume.density, res_x, res_y)

    camera.position(plume.res_x / 2, plume.res_y / 2, 240)
    camera.lookat(plume.res_x / 2, plume.res_y / 2, 0)
    scene.set_camera(camera)

    # scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices=V,
               indices=F,
               per_vertex_color=C,
            #    show_wireframe=True
               )

    canvas.scene(scene)
    window.show()