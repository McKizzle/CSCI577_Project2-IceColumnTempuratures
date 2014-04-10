import numpy as np
import scipy.special as ss

# Constants
g = 9.81            # gravity:              m/s**2

spy = 31556926.0    # seconds in a year:    seconds / year

lrho = 911.0        # lowercase rho:        kg / m**3

C_p = 2009.0        # heat capacity:        J / (kg * K)

beta = 9.8e-8       # pressure dependence 
                    #  of melting point:    K / Pa

k = 2.1             # themal diffusivity:   W / (m * K)

u_s = 90.0          # h surface velocity:   m / a

u_b = 0.0           # h basal velocity:     m / a

aacc = -0.5         # surface mass balance: m / a

pz_spx = 0.7        # surface slope of ice: degrees

lmbda = 7.0e-3      #temp. lapse rate:      degrees / m

# horizonatal temperature gradient ?? -- this is confusing me!

Q_geo = 3.2e-2      #geothermal heat flow:  W / m**2


def TAnalytic(z,w,H,ubar,alpha,lamb,Qgeo,Ts):
    """
        z     = vector of z coordinate postions such that 0 is surface and bed is negative thickness
        w     = vertical velocity in meters per second. also a vector of the same size as z
        H     = ice thickness
        ubar  = vertical average of the horizontal speed in meters per second.
        alpha = surface slope as a ratio of rise to run.
        lamb  = elevational lapse rate
        Qgeo  = geothermal heat flow
        Ts    = Surface temperature (mean annual)
        OUTPUT:
        T     = a vector of temperatures
    """
    assert len(z) == len(w), "z and w do not have the same length."
    assert z[0] == 0 and z[-1] < z[0], "Reverse array such that the top is the first element"
    k    = 2.39     # Thermal conductivity of ice W/m/K
    rhoi = 911      # Density of ice kg/m^3
    cp   = 2009     # Heat Capacity of ice J/K/kg
    spy  = 31556926 # Seconds per year
    w = w * spy
    ubar = ubar * spy
    xphi = np.sqrt((w[0] - w[-1]) / (2 * H * (k/(rhoi * cp)*spy)))
    coef = 2. * ubar * alpha * lamb * H / (w[0] - w[-1])
    
    T =  Ts - Qgeo / (k * xphi) * np.sqrt(np.pi)/2. * \
        (ss.erf(xphi*H) - ss.erf(xphi * (-z))) \
        + coef * (ss.dawsn(xphi * H) - ss.dawsn(xphi * (-z)))

    return T[-1::-1]  # Be careful with this, your coordinate system may be different.

def upa2ups(U):
    """ Converts units per anon to units per second """
    return U / spy

def deg2rad(deg):
    return deg / 360.0 * np.pi * 2

def deg2ratio(deg):
    return np.sin(deg2rad(deg)) / np.cos(deg2rad(deg))

def theta_PMP(Z):
    """ Pressure melting point of ice at bed 
        :param Z: numpy array of depths.  
    """
    z_s = np.amax(Z)
    z_b = np.amin(Z)
    return beta * lrho * (z_s - z_b)

def u_z(Z):
    """ Horizontal ice velocity at depth
        :param Z: numpy array of depths
    """
    return u_s * sigmoid_z(Z)**4

def w_z(Z):
    """ Vertical ice velocity 
        :param Z: numpy array of depths
    """
    return (aacc + u_s * pz_spx) * sigmoid_z(Z)

def sigmoid_z(Z):
    """ Calcualte the re-scaled vertical coordinate (normalize) """
    assert type(Z) is np.ndarray, "Cost of a non numpy array? One application crash. Cost of a numpy array? Priceless."

    z_s = np.amax(Z)
    z_b = np.amin(Z)

    assert z_s > z_b, "Invalid surface and base elevations z_s and z_b found in sigmoid_z"

    s_z = Z - z_s
    s_z = Z / (z_s - z_b)
    
    assert len(s_z) == len(Z), "The length of s_z and Z are not the same."
    return s_z

