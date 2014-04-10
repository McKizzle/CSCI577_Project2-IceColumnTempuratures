import numpy as np
import matplotlib.pyplot as plt
import IceMath as IM

class IceColumn:
    def __init__(self, z_range=[0,99], theta_0 = np.zeros(100), name=None):
        """
            :param z_range: The depth range
            :param theta_0: The initial tempurature values. 
            :param name: The name fo the IceColumn model. 
        """
        self.Z = np.linspace(z_range[0], z_range[-1], len(theta_0))
        self.Theta = [theta_0] # set the initial tempurature range. 
        self.name = name

    def plot(self, other_ice_columns = None):
        """ 
            Plots the last state of  ice column and if needed plots other ice 
            columns for comparison. 

            :param other_ice_columns: a list of IceColumn objects (will also 
                plot their last states).
                for the moment only a maximum of five can be plotted at a time.
        """
        line_styles = ['-', '-', '-', '-']
        marker_styles = ['o', 'o', 'o', 'o']
        colors = ['b', 'g', 'r', 'c']

        plt0, = plt.plot(self.Theta[-1], self.Z, '-sk')
        plt.xlabel("Tempurature (C)")
        plt.ylabel("Depth (m)")

        plts = [plt0]
        lbls = [self.name]
        if other_ice_columns is not None:
            for i in range(0, len(other_ice_columns)): 
                plttmp, = plt.plot(other_ice_columns[i].Theta[-1], other_ice_columns[i].Z,
                    line_styles[i] + marker_styles[i] + colors[i]
                )
                plts.append(plttmp)
                lbls.append(other_ice_columns[i].name)
        
        plt.legend(plts, lbls)
        plt.show() 

class Boundary:
    def __init_(self):
        self.boundaries = {}

    def add_boundary(self, x, y):
        pass

def analytic():
    data = borehole_data()
    data[:,0] = -data[:,0] + 1
    data = data[::-1]
    Z = data[:, 0]
    w = IM.upa2ups(IM.w_z(Z))
    H = np.amax(Z) - np.amin(Z)
    ubar = np.average(w)
    alpha = IM.deg2ratio(IM.pz_spx)
    lamb = IM.lmbda
    Qgeo = IM.Q_geo
    Ts = 0.0

    theta = IM.TAnalytic(Z, w, H, ubar, lamb, alpha, Qgeo, Ts) 
    return IceColumn([Z[0], Z[-1]], theta, "Analytical Data")

def borehole():
    data = borehole_data()
    return IceColumn([-data[0, 0], -data[-1, 0]], borehole_data()[:,1], "Borehole Data")

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


