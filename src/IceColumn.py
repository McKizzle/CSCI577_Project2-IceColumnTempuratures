import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.integrate as scint
import scipy as scpy
import scipy.sparse as sprs

spy = 31556926.0    # seconds in a year:    s / a

def borehole():
    bh_dat = borehole_data()
    bh_dat = borehole_data()
    bh_dat = bh_dat[-1::-1]
    bh_dat[:, 0] = -bh_dat[:,0]
    bh_ice = IceColumn(name="Borehole Data")
    bh_ice.Theta = [bh_dat[:,1]]
    bh_ice.Z = bh_dat[:,0]
    return bh_ice

# Creates a default simulation starting slate. 
def simulation_with_borehole(aacc=-0.5, u_b=0.0):
    """ Creates a starting simulation state. The simulation uses the borehole
        data surface and basal tempuratures for its tempuratures. The 
        depth and number of nodes in the borehole data determines the 'z' step
        and the maximum depth in the simulation. 
    """
        
    # Initialize a blank slate. 
    bh_dat = borehole_data()
    bh_dat = bh_dat[-1::-1]
    bh_dat[:, 0] = (-bh_dat[:,0]) + 1.0
    theta_s = bh_dat[0, 1]
    theta_b = bh_dat[-1, 1]
    N = bh_dat.shape[0]
    z_depth = np.min(bh_dat[:,0])
    simu_ice = IceColumn(depth=z_depth, N=N, name="Simulated Ice", aacc=aacc, u_b=u_b)

    # Set the surface and basal temps
    simu_ice.theta_s = bh_dat[ 0, 1]
    simu_ice.theta_b = bh_dat[-1, 1]
    #simu_ice.Theta = simu_ice.TAnalytic()
    simu_ice.Theta[-1][0]  = bh_dat[ 0, 1]
    simu_ice.Theta[-1][-1] = bh_dat[ 0, 1]

    return simu_ice

def borehole_data():
    return np.array(
        [[ 801.       ,   -1.4404231],
        [ 741.       ,   -3.3368846],
        [ 721.       ,   -4.3305769],
        [ 701.       ,   -5.3550385],
        [ 681.       ,   -6.4179615],
        [ 661.       ,   -7.3731923],
        [ 641.       ,   -8.3976538],
        [ 621.       ,   -9.3298077],
        [ 601.       ,   -9.9696538],
        [ 581.       ,  -10.947962 ],
        [ 561.       ,  -11.572423 ],
        [ 541.       ,  -12.012269 ],
        [ 521.       ,  -12.652115 ],
        [ 501.       ,  -12.630423 ],
        [ 481.       ,  -12.654885 ],
        [ 461.       ,  -13.817808 ],
        [ 441.       ,  -13.834577 ],
        [ 421.       ,  -12.9975   ],
        [ 401.       ,  -13.429654 ],
        [ 381.       ,  -13.261808 ],
        [ 361.       ,  -13.863192 ],
        [ 341.       ,  -13.472269 ],
        [ 321.       ,  -13.565962 ],
        [ 301.       ,  -13.782731 ],
        [ 281.       ,  -13.168731 ],
        [ 261.       ,  -13.131654 ],
        [ 241.       ,  -12.963808 ],
        [ 221.       ,  -12.095962 ],
        [ 201.       ,  -12.0435   ],
        [ 181.       ,  -12.152577 ],
        [ 161.       ,  -11.577038 ],
        [ 141.       ,  -11.3015   ],
        [ 121.       ,  -11.279808 ],
        [ 101.       ,  -10.5735   ],
        [  81.       ,  -10.074885 ],
        [  61.       ,   -9.6993462],
        [  41.       ,   -9.5238077],
        [  21.       ,   -9.4867308],
        [   1.       ,   -9.6111923]]
    )


def deg2rad(deg):
    return (deg / 360.0) * np.pi * 2

def deg2ratio(deg):
    return np.sin(deg2rad(deg)) / np.cos(deg2rad(deg))

class IceColumn:
    def __init__(self, depth=-800.0, N = 50, name="Default Simulation", u_b=0.0, aacc=-0.5):
        """
            Convention: The surface of the glaciar is always zero. Depths below
                the surface are always negative. 

            :param depth: Expects the depth to be a negative value. If the value is positive 
                then it will be converted a negative value. 
            :param N: The number of nodes in the the simulation. 
            :param name: The name of the IceColumn model. 
        """
        if depth > 0:
            depth = -1 * depth

        self.Z = np.linspace(0.0, depth, N)
        self.Theta = [np.zeros(N)] # set the initial tempurature range. 
        self.theta_s = 0.0 # degrees celcius
        self.theta_b = 0.0 # degrees celcius
        self.name = name
        self.N = N
        self.dz = np.abs(self.Z[0] - self.Z[1])
        
        # Set the default properties of the ice column.
        self.g = -9.81          # earth gravity:        m/s**2
        self.lrho = 911.0       # lowercase rho:        kg / m**3
        self.cp = 2009.0        # heat capacity:        J / (kg * K)
        self.beta = 9.8e-8      # pressure dependence 
                                #  of melting point:    K / Pa
        self.k = 2.1            # themal diffusivity:   W / (m * K)
        self.u_s = 90.0         # horizontal surface velocity:   m / a
        self.u_b = u_b          # horizontal basal velocity:     m / a
        self.aacc = aacc        # surface mass balance: m / a
        self.pz_spx = -0.7       # surface slope of ice: degrees
        self.lmbda = 7.0e-3     #temp. lapse rate:      degrees / m
        # horizonatal temperature gradient ?? -- this is confusing me!
        self.Q_geo = -3.2e-2     #geothermal heat flow:  W / m**2
        self.Q_f = self.frictional_heat()
        self.Q_total = self.Q_geo + self.Q_f

        # calculate the pressure melting point for the ice sheet.
        self.theta_pmp = self.pressure_melting_point() #self.beta * self.lrho * (self.z_s - self.z_b)

        # calcualte sigma(z)
        self.sig_z = self.rescaled_vertical_coord()

        # calcualte w(z)
        self.w_z  = self.vertical_velocity()

        # calculate phi(z)
        self.phi_z = self.deformation_heat_sources()

        # Calcualte the horizontal ice velocities at depths
        self.u_z = self.horizontal_velocity()
 
    def sim_time_settings(self, a_0=0, a_n=10000, da=100):
        """ Set the simulation time settings in years. Defaults to ten years.
            :param a_0: starting year
            :param a_n: stopping year
            :param da:  time step in years.
        """
        self.t_0 = a_0 * spy
        self.t_n = a_n * spy
        self.dt  = da  * spy

    def sim_params(self, u_b, aacc):
        """ Set the simulation parameters
            :param u_b: the basal velocity in m / year
            :param aacc: the accumulation rate in m / year
        """
        self.aacc = aacc
        self.u_b = u_b

        #update everything to reflect the changes. 
        self.Q_f = -self.frictional_heat() / spy
        self.Q_total = self.Q_geo + self.Q_f

        # calculate the pressure melting point for the ice sheet.
        self.theta_pmp = self.pressure_melting_point() #self.beta * self.lrho * (self.z_s - self.z_b)

        # calcualte sigma(z)
        self.sig_z = self.rescaled_vertical_coord()

        # calcualte w(z)
        self.w_z  = self.vertical_velocity()

        # calculate phi(z)
        self.phi_z = self.deformation_heat_sources()

        # Calcualte the horizontal ice velocities at depths
        self.u_z = self.horizontal_velocity() 
    
    def simulate(self):
        """ Start the simulation. Remember to call sim_time_settings to 
            setup the simulation params. 
        """

        # Initialize RHS variables. 
        D = self.diffusion_matrix()
        D[-1, -1] = self.Q_total / spy
        A = self.advection_matrix() / spy
        w_z = self.w_z
        u_z = self.u_z / spy
        
        intgr = scint.ode(self.rhs) 
        intgr.set_integrator('vode', method='bdf')
        intgr.set_f_params(D, A, w_z, u_z)
        intgr.set_initial_value(self.Theta[-1], self.t_0)
        print "Accumulation Rate m/a: %f", self.aacc
        print "Basal Velocity m/a: %f", self.u_b
        print "Simulating..."
        while(intgr.t < self.t_n):
            self.Theta.append(intgr.integrate(intgr.t + self.dt))
        print "Done!"

    def diffusion_matrix(self):
        """ Construct the diffusion matrix operator """ 
        k  = self.k
        lrho  = self.lrho
        Cp = self.cp 
        dz = self.dz

        R = sprs.lil_matrix((self.N, self.N)) 
        R.setdiag(np.ones(self.N), k=1)
        R.setdiag(np.ones(self.N) * - 2.0)
        R.setdiag(np.ones(self.N), k=-1)
        R = R * k / (lrho * Cp * dz ** 2)
        R[0,:] = np.zeros(self.N)
        R[-1,:] = np.zeros(self.N)

        return R

    def advection_matrix(self):
        """ Construct the advection matrix operator """
        M = sprs.lil_matrix((self.N, self.N))
        dz = self.dz
        v = self.aacc

        # Upwinding
        """ Upwind formula from the wiki """
        A = sprs.lil_matrix((self.N, self.N)) 
        if v < 0:
            A.setdiag(np.ones(self.N) * -1.0)
            A.setdiag(np.ones(self.N), k=1)
        elif v == 0:
            A = A * 0
        elif v > 0:
            A.setdiag(np.ones(self.N) * -1.0, k=-1)
            A.setdiag(np.ones(self.N))

        A[0,:] = np.zeros(self.N)
        A[-1,:] = np.zeros(self.N)
        A = A / (2 * dz)
        
        # Set up the final row so that it can handle the base of the system. 
        B = sprs.lil_matrix((self.N, self.N))
        B.setdiag(np.ones(self.N) *  3.0, k= 0)
        B.setdiag(np.ones(self.N) * -4.0, k=-1)
        B.setdiag(np.ones(self.N)       , k=-2)
        B = B / (2 * dz)
        A[-1,:] = B[-1, :]

        return A 

    def correct_boundary(self, theta, M):
        """ Correct boundary takes in a matrix a vector of theta values
            and a matrix M. It assumes that is a diffusion matrix. If the
            basal temp has reached its melting point remove the geo thermal
            heat as a source. Return a tuple of the modified theta and 
            matrix. 
            :param theta: numpy array of tempuratures.
            :param M: the diffusion matrix. 
        """
        if theta[-1] >= self.theta_pmp:
            theta[-1] = self.theta_pmp
            M[-1, -1] = 0.0
        
        return (M, theta)

    def rhs(self, t, y, D, A, w_z, u_z): 
        # remove the geothermal heatflow once the tempurature is at theta_pmp
        D, y = self.correct_boundary(y, D) 

        ptpt_d = (D * y)
        pz_spx = np.sin(deg2rad(self.pz_spx))
        ptpt_special = self.lmbda * pz_spx * u_z
        ptpt_a = (A * y * w_z)
        ptpt_source =  -self.phi_z / (self.lrho * self.cp) / spy
        ptpt = ptpt_d + ptpt_special + ptpt_a + ptpt_source

        ptpt[0] = 0.0 # lock the surface temp.
        return ptpt

    def check_end(self, ptpt, mssg):
        if ptpt[-1] < 0.0:
            print "---->", mssg
            print ptpt
            exit()


    def TAnalytic(self):
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
            T     = a vector of temperatures (almost)
        """
        z = self.Z
        w = self.w_z / spy
        H = self.Z[0] - self.Z[-1]
        ubar  = self.u_s / spy #np.average([self.u_s, self.u_b]) / spy
        alpha = deg2ratio(self.pz_spx)
        lamb  = self.lmbda
        Qgeo  = self.Q_geo
        Ts    = self.theta_s 
        k    = self.k    # Thermal conductivity of ice W/m/K
        rhoi = self.lrho # Density of ice kg/m^3 
        cp   = self.cp   # Heat Capacity of ice J/K/kg
        w    = w * spy
        ubar = ubar * spy 
        xphi = np.sqrt(np.abs(w[0] - w[-1]) / (2 * H * (k/(rhoi * cp)*spy)))
        coef = 2. * ubar * alpha * lamb * H / (w[0] - w[-1])
        
        T = Ts - Qgeo / (k * xphi) * np.sqrt(np.pi)/2. * \
            (ss.erf(xphi*H) - ss.erf(xphi * (-z))) \
            + coef * (ss.dawsn(xphi * H) - ss.dawsn(xphi * (-z)))

        return [T[-1::-1]]  # Be careful with this, your coordinate system may be different.

    def pressure_melting_point(self):
        """ Pressure melting point of ice at bed 
                $ \Theta_{PMP} = \beta \rho g(z_s - z_b) $
            :param Z: numpy array of depths.  
        """ 
        return self.beta * self.lrho * self.g * (self.Z[0] - self.Z[-1])

    def horizontal_velocity(self):
        """ Horizontal ice velocity at depth
            $ u(z) = u_s \sigma(z)^4 $ 

            return a numpy array of velocities
        """
        #return self.u_s * self.sigmoid_z()**4
        return (self.u_s - self.u_b) * self.rescaled_vertical_coord()**4 + self.u_b

    def rescaled_vertical_coord(self):
        """ Calcualte the re-scaled vertical coordinate (normalize)
            $ \sigma(z) = \frac{z - z_b}{z_s - z_b} $ 

            return a numpy array of the recaled vertical coordinates. 
        """ 
        return (self.Z - self.Z[-1]) / (self.Z[0] - self.Z[-1])

    def vertical_velocity(self):
        """ Vertical ice velocity  
            $ w(z) = (\dot{a} + u_s \frac{\partial z_s}{\partial x})  \sigma(z) $

            return a numpy array of the rescaled values. 
        """
        pz_spx = np.sin(deg2rad(self.pz_spx))
        return (self.aacc + self.u_s * pz_spx) * self.rescaled_vertical_coord()

    def vertical_shear(self):
        """ Calculates the vertical shear.

            $ frac{\partial u}{\partial z} = 4(u_s - u_b)\sigma(x)^3(z_s - z_b) $
        """
        return 4.0 * (self.u_s - self.u_b) * self.rescaled_vertical_coord()**3 / (self.Z[0] - self.Z[-1])

    def deformation_heat_sources(self):
        """ Calcualte the heat sources from the deformation of the ice. 

            $ \rho g (z_s - z) \frac{\partial u}{\partial z} \frac{\partial z_s}{\partial x} $

            return a numpy array of all of the deformation values. 
        """
        pz_spx = np.sin(deg2rad(self.pz_spx))
        phi_z = self.lrho * self.g * \
            (self.Z[0] - self.Z) * self.vertical_shear() * pz_spx 

        return phi_z

    def frictional_heat(self):
        """ Calculate the frictional heat at the base of the glacier"""
        return self.lrho * self.g * (self.Z[0] - self.Z[-1]) * np.sin(deg2rad(self.pz_spx)) * self.u_b

    def horizontal_tempurature_gradient(self):
        """ Calculate the horizontal tempurature gradient. 
            $\lambda \frac{\partial z_s}{\partial x}
        """

        return self.lmbda * np.sine(deg2rad(self.pz_spx))

    def plot(self, other_ice_columns = None, t=-1):
        """ 
            Plots the last state of  ice column and if needed plots other ice 
            columns for comparison. 

            :param other_ice_columns: a list of IceColumn objects (will also 
                plot their last states).
                for the moment only a maximum of five can be plotted at a time.
        """
        t = int(t)

        line_styles = ['-', '-', '-', '-']
        marker_styles = ['o', 'o', 'o', 'o']
        colors = ['b', 'g', 'r', 'c']

        plt0, = plt.plot(self.Theta[t], self.Z, '-sk')
        plt.xlabel("Tempurature (C)")
        plt.ylabel("Depth (m)")

        plts = [plt0]
        lbls = [self.name]
        if other_ice_columns is not None:
            for i in range(0, len(other_ice_columns)): 
                plttmp, = plt.plot(other_ice_columns[i].Theta[t], other_ice_columns[i].Z,
                    line_styles[i] + marker_styles[i] + colors[i]
                )
                plts.append(plttmp)
                lbls.append(other_ice_columns[i].name)
        
        plt.legend(plts, lbls)

class Icetimizer():
    def __init__(self, simulator=simulation_with_borehole(), golden=borehole()):
        """ Serves as a wrapper function for scipy's fmin function. 
            :param simulated: An IceColumn object that will be simulated.
            :param golden: An IceColumn object that has the real data. 
        """
        self.simulator = simulator
        self.golden = golden
        self.u_b = 10.0
        self.aacc= -.5
        self.fopt = None
        self.iters= None
        self.funcalls = None
        self.warnflag = None
        self.allvects = None
        self.xtol=.001
        self.ftol=.001

    def results_as_dict(self):
        results = {
            "Basal Velocity (m/a)"      : self.u_b,
            "Accumulation Rate (m/a)"   : self.aacc,
            "Minimum Error"             : self.fopt,
            "Iterations"                : self.iters,
            "Function Calls"            : self.funcalls,
            "Warning Flag"              : self.warnflag,
            "Error at Each Iteration"   : self.allvects,
            "Xtol"                      : self.xtol,
            "Ftol"                      : self.ftol
        }

        return results

    def optimize(self, u_b0=10.0, aacc_0=-.5):
        """ Calls scipy's optimize function on the simulation. """
        import scipy.optimize as sopty

        xopt, fopt, iters, funcalls, allvecs = sopty.fmin(self.to_optimize, [u_b0, aacc_0], 
            xtol=self.xtol, ftol=self.ftol, full_output=1)

        self.u_b  = float(xopt[0])
        self.aacc = float(xopt[1])
        self.fopt = float(fopt)
        self.iters= int(iters)
        self.funcalls = int(funcalls)
        self.allvects = allvecs

    def to_optimize(self, params):
        """ Function that is passed into fmin. Serves as a wrapper for the
            simulation
        """
        self.simulator.sim_params(params[0], params[1])
        self.simulator.simulate()

        return self.error(self.golden.Theta[-1], self.simulator.Theta[-1])



    def error(self, golden, simulated):
        """ Square root of the sum squares of two numpy vectors """
        return np.sqrt(np.sum((golden - simulated)**2))


