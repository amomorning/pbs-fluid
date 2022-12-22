import taichi as ti
import click
from Plume2d import Plume2d
from Renderer import Renderer
from Solid import SolidBox, SolidSphere


@click.command()
@click.option('--gpu/--no-gpu', default=True, help='Use cuda if gpu is available', show_default=True)
@click.option('-x', '--res_x', default=256, help='Resolution x', show_default=True)
@click.option('-y', '--res_y', default=256, help='Resolution y', show_default=True)
@click.option('--dx', default=1, help='dx', show_default=True)
@click.option('--dt', default=0.02, help='dt', show_default=True)
@click.option('--accuracy', default=1e-5, help='accuracy', show_default=True)
@click.option('--n_iters', default=100, help='number of iterations', show_default=True) 
@click.option('--substeps', default=20, help='number of substeps', show_default=True)
@click.option('-a', '--advection', default='MAC', type=click.Choice(['MAC', 'SL', 'FLIP'], case_sensitive=False), help='advection method', show_default=True)
@click.option('-e', '--interpolation', default='cerp', type=click.Choice(['cerp', 'bilerp'], case_sensitive=False), help='interpolation method', show_default=True)
@click.option('-i', '--integration', default='rk3', type=click.Choice(['rk3', 'euler'], case_sensitive=False), help='integration method', show_default=True)
@click.option('-s', '--solver', default='GS', type=click.Choice(['GS', 'MIC'], case_sensitive=False), help='solver method', show_default=True)
@click.option('-r', '--reflection', default=False, is_flag=True, help='boolean flag to use reflection')
@click.option('-w', '--wind', default=False, is_flag=True, help='boolean flag to add wind')
@click.option('-b', '--body', default=False, is_flag=True, help='boolean flag to add default solid bodies')
@click.option('-o', '--output', default=False, is_flag=True, help='boolean flag for output step png to output folder')
def main(gpu, res_x, res_y, dx, dt, accuracy, n_iters, 
        advection, interpolation, integration, solver, 
        reflection, wind, body, output, substeps):
    if gpu: ti.init(arch=ti.cuda)
    else: ti.init(arch=ti.cpu)

    bodies = [
        SolidBox(.5, .6, .7, .1, -.7),
        SolidSphere(.7, .3, .15),
        ] if body else []

    args = {
        'res_x': res_x,
        'res_y': res_y,
        'dx': dx,
        'dt': dt,
        'accuracy': accuracy,
        'poisson_iters': n_iters,
        'wind': wind,
        'advection': advection,             
        'interpolation': interpolation,    
        'integration': integration,        
        'solver': solver,
        'reflection': reflection,           
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

    frame = 0

    while window.running:

        for _ in range(substeps):
            plume.substep()
        renderer.render(plume.density, ti.Vector([0,0,0]), ti.Vector([1,1,1]))


        camera.position(plume.res_x * dx / 2, plume.res_y * dx / 2, 300 * res_x / 256)
        camera.lookat(plume.res_x * dx / 2, plume.res_y * dx / 2, 0)
        scene.set_camera(camera)

        scene.ambient_light((1, 1, 1))
        scene.mesh(vertices=renderer.V,
                indices=renderer.F,
                per_vertex_color=renderer.C,
                #    show_wireframe=True
                )

        canvas.scene(scene)
        
        if output:
            window.save_image(f"./output/{advection}_{interpolation}_{integration}_{solver}_{reflection}_{frame:05d}.png")
        window.show()

        frame += 1

if __name__ == "__main__":
    main()
