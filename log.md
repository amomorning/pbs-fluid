# Issues Recorded Here

## Smoke not symmetric
The smoke produced is not symmetric, since the exercise also doesn't produce symmetric smoke. Unknown reason. 

## Boundary conditions
Boundary conditions seem weird in the exercise. 

In [cg_fluid](cg_fluid/CG_Fluid.py), the velocity boundary conditions are:
- Zero at 4 corners
- Bounce back in normal directions at four walls

The pressure boundary condition is the same, we copy the second to last row/col to the border. 