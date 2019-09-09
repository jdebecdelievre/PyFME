from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
from json import dump, load
import pandas as pd
from pyfme.utils.change_euler_quaternion import quatern2euler, euler2quatern, change_basis
import quaternion as npquat
from pyfme.utils.coordinates import hor2body, body2hor
import matplotlib.pyplot as plt
from matplotlib import rc as rcparams
from copy import deepcopy

DEFAULT_TITLE = {
    'alpha': 'Angle of attack (deg)',
    'beta': 'Angle of sideslip (deg)',
    'phi': 'Roll angle (deg)',
    'theta': 'Pitch angle (deg)',
    'psi': 'Yaw angle (deg)',
    'attitude': 'Euler angle (deg)',
    'time': 'Time (s)',
    'p': 'p (rad/s)',
    'q': 'q (rad/s)',
    'r': 'r (rad/s)',
    'phat': '$\\hat{p}$',
    'qhat': '$\\hat{q}$',
    'rhat': '$\\hat{r}$',
    'omega': 'rotation rates (rad/s)',
    'quat': 'quaternion',
    'q0': 'q0',
    'qx': 'qx',
    'qy': 'qy',
    'qz': 'qz',
    'velocity': 'velocity (m/s)',
    'u': 'u (m/s)',
    'v': 'v (m/s)',
    'w': 'w (m/s)',
    'x_e': 'x (m)',
    'y_e': 'y (m)',
    'z_e': 'z (m)',
    'position': 'position (m)',
    'earth_velocity' : 'velocity (m/s)',
    'V':'Airspeed (m/s)'
}
DEGVARS = ['alpha', 'beta', 'phi', 'theta', 'psi', 'attitude']

DEFAULT_RC = dict(
    legend = dict(fontsize=15),
    axes= dict(labelsize=20, titlesize=20),
    figure = dict(figsize=(16,8))
)

class State():
    def __init__(self, dimension, info, state_vec, time, from_json, N):
        """
        Creates the vector.
        Parameters
        ----------
        state_vec: only important attribute
        time: (optionnal) a vector with the times of each state
        from_json: load state from a json file
        N: when creating an object and filling in state_vec later on, state_vec can be initialized to the right
        dimension
        """
        self.info = info
        self.dimension = dimension

        # Make room for N states if N is specified
        if state_vec is None:
            N = 1 if N is None else N
            state_vec =  np.ones((N,dimension)) * float('nan')
        if time is None:
            self.time = (np.ones(1) if N is None else np.ones((N,1))) * float('nan')
        else:
            self.time = time
        # transpose if the state is [n,m], we want it [m,n] so that state_vec[1] is one state example
        if state_vec.ndim > 1:
            if state_vec.shape[1] != dimension:
                state_vec = state_vec.T
        else:
            state_vec = np.expand_dims(state_vec, axis=0)
        self.vec = state_vec*1.

        # Load from json if provided
        if from_json != None:
            self.load_from_json(from_json)

    def save_to_json(self, filename):
        state = {k: getattr(self, k) for k in state.info}

        with open(filename, 'w') as f:
            dump(state, f)

    def load_from_json(self, filename):
        self.vec = np.zeros(self.dimension)
        with open(filename, 'r') as f:
            state = load(f)
        for c_name, c_fun in state.items():
            try:
                setattr(self, c_name, c_fun)
            except AttributeError:
                print(f"Wrong argument name {c_name}")

    def to_pandas(self):
        df = pd.DataFrame(self.vec, columns=self.info) 
        if np.isnan(self.time).any():
            return df
        else:
            df['time'] = self.time
            return df

    @property
    def V(self):
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def N(self):
        # number of time steps are stored in this state object (for vectorization)
        return self.vec.shape[0]


# Descriptor for properties in state
class StateProperty(object):
    def __init__(self,slice):
        self.slice = slice

    def __get__(self, instance, owner):
        return instance.vec.T[self.slice].T

    def __set__(self, instance, value):
        instance.vec[:, self.slice] = value*1.


class BodyAxisState(State):
    """
    This class only contains an attribute vec that contains a state=np.array(12).
     It can also contain a set of states. Then np.array(m,12), such that state.vec[0] is a 12-dim state.
     One can get a vector of a particular value for all states saved by calling state.value.
    """

    # Define State Properties
    x_e = StateProperty(0)
    y_e = StateProperty(1)
    z_e = StateProperty(2)
    position = StateProperty([0,1,2])

    phi = StateProperty(3)
    theta = StateProperty(4)
    psi = StateProperty(5)
    attitude = StateProperty([3,4,5])

    u = StateProperty(6)
    v = StateProperty(7)
    w = StateProperty(8)
    velocity = StateProperty([6,7,8])

    p = StateProperty(9)
    q = StateProperty(10)
    r = StateProperty(11)
    omega = StateProperty([9,10,11])

    def __init__(self, state_vec=None, time=None, from_json=None, N=None):
        
        # Define infos and dimensions
        info = ["x_e","y_e","z_e", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]
        dimension = 12
        super().__init__(dimension, info, state_vec, time, from_json, N)

    def __repr__(self):
        rv = (
            "Aircraft State \n"
            f"position : {self.position} m\n"
            f"attitude : {self.attitude} rad\n"
            f"velocity : {self.velocity} m/s\n"
            f"omega : {self.omega} rad/s\n"
            "Vector of size 12."
        )
        return rv


class BodyAxisStateQuaternion(State):
    # Define State Properties
    x_e = StateProperty(0)
    y_e = StateProperty(1)
    z_e = StateProperty(2)
    position = StateProperty([0,1,2])

    q0 = StateProperty(3)
    qx = StateProperty(4)
    qy = StateProperty(5)
    qz = StateProperty(6)

    u = StateProperty(7)
    v = StateProperty(8)
    w = StateProperty(9)
    velocity = StateProperty([7,8,9])

    p = StateProperty(10)
    q = StateProperty(11)
    r = StateProperty(12)
    omega = StateProperty([10,11,12])

    def __init__(self, state_vec=None, time=None, from_json=None, N=None, aircraft=None, 
                default_titles=DEFAULT_TITLE, deg_vars=DEGVARS, rcParams=DEFAULT_RC):
        # Define infos and dimensions
        info = ["x_e","y_e","z_e", "q0","qx", "qy", "qz", "u", "v", "w", "p", "q", "r"]
        dimension = 13
        if N is None:
            if state_vec is None:
                N = 1
            else:
                N = state_vec.shape[0]
        super().__init__(dimension, info, state_vec, time, from_json, N)
        self.aircraft = aircraft
        self.default_titles = default_titles
        self.deg_vars = deg_vars
        self.rcParams = rcParams


    @property
    def quaternion(self):
        return npquat.as_quat_array(self.vec.T[[3,4,5,6]].T)
    @quaternion.setter
    def quaternion(self, value):
        if type(value) == npquat.quaternion  or type(value[0]) == npquat.quaternion:
            self.vec[:, 3:7] = npquat.as_float_array(value)
        elif type(value) == np.ndarray:
            self.vec[:, 3:7] = value
        
    @property
    def phi(self):
        return self.attitude[:,0]
    @property
    def theta(self):
        return self.attitude[:,1]
    @property
    def psi(self):
        return self.attitude[:,2]

    @property
    def attitude(self):
        return quatern2euler(npquat.as_float_array(self.quaternion))
    @attitude.setter
    def attitude(self, value):
        # set the quaternion vector
        if type(value) == list:
            value = np.array(value)
        if value.ndim == 1:
            value = np.expand_dims(value, axis=0)
        self.quaternion = euler2quatern(value)

    @property
    def alpha(self):
        return np.arctan2(self.w, self.u)
    @property
    def beta(self):
        return np.arcsin(self.v / self.V)
    @property
    def qhat(self):
        assert self.aircraft is not None, "The state object must have an argument named aircraft."
        return self.q * self.aircraft.chord / 2 / self.V
    @property
    def phat(self):
        assert self.aircraft is not None, "The state object must have an argument named aircraft."
        return self.p * self.aircraft.span / 2 / self.V
    @property
    def rhat(self):
        assert self.aircraft is not None, "The state object must have an argument named aircraft."
        return self.r * self.aircraft.span / 2 / self.V

    @property
    def earth_velocity(self):
        return change_basis(self.velocity, self.quaternion.conjugate())
    @earth_velocity.setter
    def earth_velocity(self, value):
        # set the quaternion vector
        if type(value) == list:
            value = np.array(value)
        if value.ndim == 1:
            value = np.expand_dims(value, axis=0)
        self.velocity = change_basis(value, self.quaternion)

    def plot(self, xname=None, yname=None, 
            xdata=None, ydata=None,
            xlabel=None, ylabel=None, 
            title=None, point='+',
            xdeg=None, ydeg=None,
            plotParams=None, hist=False, nbins=100, fig=None):
        
        # initialize fig
        plotParams = self.rcParams if plotParams is None else plotParams
        for k in plotParams:
            rcparams(k, **plotParams[k])
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0]

        # get data
        if xdata is None:
            xdata = getattr(self, xname)
        if ydata is None and not hist:
            ydata = getattr(self, yname)
        
        # convert radians to degrees if necessary
        if xdeg is None:
            xdeg = True if xname in self.deg_vars else False
        if ydeg is None:
            ydeg = True if yname in self.deg_vars else False
        xdata = xdata*180/np.pi if xdeg else xdata
        ydata = ydata*180/np.pi if ydeg else ydata

        # set up default labels titles
        if xlabel is None:
            xlabel = '' if xname is None else self.default_titles[xname]
        if ylabel is None and not hist:
            ylabel = '' if yname is None else self.default_titles[yname]
        
        # plot
        if not hist:
            ax.plot(xdata, ydata, point)
            ax.set_ylabel(ylabel)
        else:
            ax.hist(xdata, nbins)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        return fig, ax
    
    def __getitem__(self, idx):
        S = deepcopy(self)
        S.time = self.time[idx]
        S.vec = self.vec[idx]
        return S

    def __repr__(self):
        rv = (
            "Aircraft State with quaternions.\n"
            f"position : {self.position} m\n"
            f"quaternion : {self.quaternion} \n"
            f"velocity : {self.velocity} m/s\n"
            f"omega : {self.omega} rad/s\n"
            "Vector of size 13. Attitude (i.e. euler angles) is a computed property."
        )
        return rv


def copyStateValues(state1, state2):
    for attr in state1.info:
        setattr(state2, attr, getattr(state1, attr))
    return state2