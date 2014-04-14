#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import IceColumn as IC
import time as tm
import yaml

def main():
    borehole_ice = IC.borehole()

    # Initialize a blank slate. 
    bh_dat = IC.borehole_data()

    simu_ice = IC.simulation_with_borehole() 
    
    #print "New Params: "
    #simu_ice.sim_params(1.0, -.5)
    #simu_ice.sim_time_settings(a_n=5000, da=10)
    #simu_ice.simulate()
    #simu_ice.plot([borehole_ice]) 
    #animate1Dframes(simu_ice.Z, simu_ice.Theta)
    #
    #print "New Params: "
    #simu_ice.sim_params(5.0, -.5)
    #simu_ice.simulate()
    #simu_ice.plot([borehole_ice]) 

    ictmzr = IC.Icetimizer()
    ictmzr.simulator.sim_time_settings()
    ictmzr.optimize()

    results = {
        "u_b" : ictmzr.u_b,
        "aacc" : ictmzr.aacc
    }

    solution = ictmzr.simulator
    solution.plot([borehole_ice])

    with open('results.yml', 'w') as outfile:
        outfile.write(yaml.dump(results, default_flow_style=True))

def animate1Dframes(x, data):
    """ Animates a 2D array of data using pyplot. 
        :param x: the x values to plot over
        :param y: the y value 'frames' to plot at each iteration. 

        Follows example at http://www.lebsanft.org/?p=48
    """
    plt.ion() # Set the plot to animated.    
    ax1 = plt.axes()
    line, = plt.plot(data[-1], x , '-*k')

    for u in data:
        line.set_xdata(u)
        plt.draw()
        #tm.sleep(0.25)

if __name__ == '__main__':
    main()
