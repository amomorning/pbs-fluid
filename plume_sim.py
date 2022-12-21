import taichi as ti
from Plume2d import Plume2d
from Renderer import Renderer

ti.init(arch=ti.cuda)

res_x = 256
res_y = int(res_x * 1)
dx = 1
dt = 0.005 * ti.sqrt((res_x + res_y) * 0.5) / 4
accuracy = 1e-5
n_iters = 100
advection = "MAC"
interpolation = "cerp"
integration = "rk3"
solver = "GS"
reflecton = True
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
    'reflection': reflecton
}

plume = Plume2d(args)

# For rendering
renderer = Renderer(res_x, res_y, dx)


window = ti.ui.Window("Plume 2d", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

substeps = 1
frame = 0

while (window.running):
    # if plume.t_curr > 200:
    #     # Reset
    #     plume.reset()

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
    
    # window.save_image(f"./output/{advection}_{interpolation}_{integration}_{solver}_{reflecton}_{frame:05d}.png")
    window.show()

    frame += 1
