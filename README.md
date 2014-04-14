Project II: Ice Column Temperatures
======================================

Glaciers are a natural phenomena that exhibit the interest-
ing property of being able to flow even though they consist
of ice. In this project we will investigate the transfer of en-
ergy through one-dimensional section of a glacier. In this
case we will observe the temperature at different depths
subject to the internal forces, and geothermal energy re-
leased from the ground at the base of the glacier.

## Source Files
  - _main.py_ Contains code that constructs the objects, runs the simulation, and plots everything. 
  - _IceColumn.py_ Simulation logic is stored in this module.
    - _IceColumn_ a class that contains the infrastructure to simulate a glacier temperatures at a range of depths.
    - _Icetimizer_ serves as a wrapper function for `scipy.optimize.fmin`. The error that `fmin` minimizes is the sum squared error is calculated as the sum squared error. 

## Results

![Simulation Results](/img/Simulation_vs_Real.png)
![Analytical Solution Comparison](/img/bad_analytic.png)

