#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import IceColumn as IC

def main():
    borehole_ice = IC.borehole()
    analytical_ice = IC.analytic()
    borehole_ice.plot(other_ice_columns=[analytical_ice])

if __name__ == '__main__':
    main()
