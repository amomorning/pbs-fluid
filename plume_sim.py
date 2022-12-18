import taichi as ti
from Plume2d import Plume2d
from Renderer import Renderer
from Solid import SolidBox, SolidSphere

ti.init(arch=ti.cuda)

res_x = 256
res_y = int(res_x * 1)
dx = 1
dt = 0.005 * ti.sqrt((res_x + res_y) * 0.5)
accuracy = 1e-4
n_iters = 100
advection = "FLIP"
interpolation = "cerp"
integration = "rk3"
solver = "GS"
reflecton = True
bodies = [
    SolidBox(.5, .6, .7, .1, -.7),
    SolidSphere(.7, .3, .2),
    ]


reflecton = False
wind = False

args = {
    'res_x': res_x,
    'res_y': res_y,
    'dx': dx,
    'dt': dt,
    'accuracy': accuracy,
    'poisson_iters': n_iters,
    'wind': wind,
    'advection': advection,         # SL, MAC, FLIP
    'interpolation': interpolation,    # bilerp, cerp
    'integration': integration,        # euler, rk3
    'solver': solver,
    'reflection': reflecton,
    'bodies': bodies,
}

plume = Plume2d(args)

# For rendering
renderer = Renderer(res_x, res_y, dx, plume._cell)


window = ti.ui.Window("Plume 2d", (res_x*2, res_y*2),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

substeps = 10
frame = 0

while window.running:
    if frame > 500:
        break

    for _ in range(substeps):
        plume.substep()
    renderer.render(plume.density, ti.Vector([0,0,0]), ti.Vector([1,1,1]))


    camera.position(plume.res_x / 2, plume.res_y / 2, 300)
    camera.lookat(plume.res_x / 2, plume.res_y / 2, 0)
    scene.set_camera(camera)

    scene.ambient_light((1, 1, 1))
    scene.mesh(vertices=renderer.V,
               indices=renderer.F,
               per_vertex_color=renderer.C,
            #    show_wireframe=True
               )

    canvas.scene(scene)
    
    window.save_image(f"./output/{advection}_{interpolation}_{integration}_{solver}_{reflecton}_{frame:05d}.png")
    window.show()

    frame += 1
