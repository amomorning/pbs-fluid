# Physically based Fluid Simulation

Final Project for Physically based Simulation 2022.

## Usage

### Requirements

- Python 3.10.4
- Taichi 1.2.2
- click

### Installation

``` bash
pip install taichi==1.2.2
pip install click
```

### Run

``` bash
# show instructions
python plume_sim.py --help

# example usage
## default arguments with solid boundary
python plume_sim.py -b

## change some arguments
python plume_sim.py --advection=SL --interpolation=bilerp --solver=CG -b
# in short
python plume_sim.py -a SL -e bilerp -s CG -b
```

## Results

<!-- TODO -->

## Reference

- [Fluid Simulation for Computer Graphics, Second Edition](http://wiki.cgt3d.cn/mediawiki/images/4/43/Fluid_Simulation_for_Computer_Graphics_Second_Edition.pdf)
- [An Advection-Reflection Solver for Detail-Preserving Fluid Simulation](https://jzehnder.me/publications/advectionReflection/)
- [A Second-Order Advection-Reflection Solver](https://www.cse.iitd.ac.in/~narain/ar2/)
- [@tunabrain/incremental-fluids](https://github.com/tunabrain/incremental-fluids)
- [@Robslhc/WaterSim](https://github.com/Robslhc/WaterSim)
- [@taichi-dev/taichi](https://github.com/tunabrain/taichi-dev/taichi)
