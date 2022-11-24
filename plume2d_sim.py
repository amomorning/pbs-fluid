import taichi as ti
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


# Render
    
window = ti.ui.Window("Plume 2d", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    if plume.t_curr > 10:
        # Reset
        plume.reset()

    # plume.substep()
    plume.get_color()

    camera.position(plume.res_x / 2, plume.res_y / 2, 400)
    camera.lookat(plume.res_x / 2, plume.res_y / 2, 0)
    scene.set_camera(camera)

    # scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(plume.V,
               indices=plume.F,
               per_vertex_color=plume.C,
               show_wireframe=True
               )

    # Draw a smaller ball to avoid visual penetration
    canvas.scene(scene)
    window.show()