import taichi as ti
from Plume2d import Plume2d
from Renderer import Renderer

ti.init(arch=ti.cuda)

res_x = 128
res_y = int(res_x * 1)
dx = 1
dt = 0.005 * ti.sqrt((res_x + res_y) * 0.5)
accuracy = 1e-4
n_iters = 100

args = {
    'res_x': res_x,
    'res_y': res_y,
    'dx': dx,
    'dt': dt,
    'accuracy': accuracy,
    'poisson_iters': n_iters
}

plume = Plume2d(args)
plume.MAC_on = True
# plume.wind_on = True
plume.dt /= 2
plume.reflection = True

# For rendering
renderer = Renderer(res_x, res_y, dx)


window = ti.ui.Window("Plume 2d", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

substeps = 1

while window.running:
    # if plume.t_curr > 200:
    #     # Reset
    #     plume.reset()

    # for _ in range(substeps):
    plume.substep()
    renderer.render(plume.density, ti.Vector([0,0,0]), ti.Vector([1,1,1]))

    camera.position(plume.res_x / 2, plume.res_y / 2, 240)
    camera.lookat(plume.res_x / 2, plume.res_y / 2, 0)
    scene.set_camera(camera)

    scene.ambient_light((1, 1, 1))
    scene.mesh(vertices=renderer.V,
               indices=renderer.F,
               per_vertex_color=renderer.C,
            #    show_wireframe=True
               )

    canvas.scene(scene)
    
    # if plume.n_steps % 100 == 0:
    #     window.save_image(f"./output/frame{plume.n_steps % 100}.png")
    window.show()