import taichi as ti
from Simplistic import Fluid2d
from Renderer import Renderer

ti.init(arch=ti.cuda)

res_x = 128
res_y = int(res_x * 1)
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
    'poisson_iters': n_iters,
    'wind_on': False
}

# Init simulator
plume = Fluid2d(args)

# Init renderer
renderer = Renderer(res_x, res_y, dx)
render_density = True
render_divergence = False
render_voritcity = False

# Init GUI
window = ti.ui.Window("Fluid 2d", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Output parameters
substeps = 20
frame = 0

while window.running:
    # if plume.t_curr > 200:
    #     # Reset
    #     plume.reset()

    for _ in range(substeps):
        plume.substep()

    if render_density:
        renderer.render(plume.density.q, ti.Vector([1,1,1]), ti.Vector([0,0,0]))
    elif render_divergence:
        renderer.render(plume.divergence.q, ti.Vector([1,1,1]), ti.Vector([0,0,0]))
    elif render_voritcity:
        renderer.render(plume.vorticity.q, ti.Vector([1,1,1]), ti.Vector([0,0,0]))

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
    
    # Comment this line if don't want to save to png
    # window.save_image("./output/frame{:05d}.png".format(frame))

    window.show()
    frame += 1