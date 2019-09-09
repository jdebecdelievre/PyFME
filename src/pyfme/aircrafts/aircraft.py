"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Generic Aircraft
----------------

"""
from abc import abstractmethod

import numpy as np
import pdb
from pyfme.utils.coordinates import body2wind, wind2body
from copy import deepcopy as cp
from pyfme.environment import Conditions
from json import load

class Aircraft(object):

    def __init__(self, 
                mass=0, 
                inertia=np.zeros((3, 3)),
                Sw=0,
                chord=0,
                span=0,
                control_limits={}, file=None):

        # Mass & Inertia
        self.mass = mass  # kg
        self.inertia = inertia  # kg·m²

        # Geometry
        self.Sw = Sw  # m2
        self.chord = chord  # m
        self.span = span  # m

        # Controls
        self.control_limits = control_limits

        # File information
        if file:
            self.read_file(file)
    
    def read_file(self,file):
        with open(file, 'r') as f:
            caract = load(f)
        self.mass = caract['inertia']['mass']
        self.inertia = caract['inertia']['inertia']
        self.Sw = caract['geometry']['Sw']
        self.chord = caract['geometry']['chord']
        self.span = caract['geometry']['span']

    @property
    def Ixx(self):
        return self.inertia[0, 0]

    @property
    def Iyy(self):
        return self.inertia[1, 1]

    @property
    def Izz(self):
        return self.inertia[2, 2]

    @abstractmethod
    def get_controls(self, t, controls_sequence):
        raise NotImplementedError

    def calculate_forces_and_moments(self, state, conditions, controls):
        # vectorize the dirty way (just for testing)
        total_forces = np.zeros((state.N, 3))
        total_moments = np.zeros((state.N, 3))
        one_state = cp(state)
        one_control = cp(controls)
        for i in range(state.N):
            one_state.vec = state.vec[i]
            one_control.vec = controls.vec[i]
            one_cond = Conditions(*[conditions[ii][i] for ii in range(len(conditions))])
            total_forces[i, :], total_moments[i, :] = self._calculate_forces_and_moments(one_state, one_cond, one_control)
        return total_forces, total_moments

    def _calculate_forces_and_moments(self, state, conditions, controls):
        TAS = conditions.TAS
        alpha = conditions.alpha
        beta = conditions.beta
        Ft = self._calculate_thrust_forces_moments(TAS, conditions, controls)
        L, D, Y, l, m, n = self._calculate_aero_forces_moments(conditions, state, controls)
        Fa_wind = np.array([-D, Y, -L])
        Fa_body = wind2body(Fa_wind, alpha, beta)
        Fa = Fa_body
        total_forces = Ft + Fa
        total_moments = np.array([l, m, n])
        return total_forces, total_moments

    def calculate_derivatives(self, state, environment, controls, eps=1e-3):
        """
        Calculate dimensional derivatives of the forces at the vicinity of the state.
        The output consists in 2 dictionaries, one for force one for moment
        key: type of variables derivatives are taken for
        val : 3x3 np array with X,Y,Z and L,M,N as columns, and the variable we differentiate against in lines
        (u,v,w ; phi,theta,psi ; p,q,r ; x,y,z)
        """
        names = {'velocity': ['u', 'v', 'w'],
                 'omega': ['p', 'q', 'r'],
                 'acceleration': ['w_dot']}
        Fnames = ['X', 'Y', 'Z']
        Mnames = ['L', 'M', 'N']

        # F, M = self.calculate_forces_and_moments(state, environment, controls)

        # Rotation for stability derivatives in stability axis
        V = np.sqrt(state.velocity.u**2 + state.velocity.v**2 + state.velocity.w**2)
        alpha = np.arctan2(state.velocity.w, state.velocity.u)
        beta = np.arcsin(state.velocity.v / V)


        derivatives = {}
        for keyword in names.keys():
            for i in range(len(names[keyword])):
                eps_v0 = np.zeros(3)

                # plus perturb
                eps_v0[i] = eps/2
                eps_vec = wind2body(eps_v0, alpha, beta)
                state.perturbate(eps_vec, keyword)
                forces_p, moments_p = self.calculate_forces_and_moments(state, environment, controls)
                forces_p = body2wind(forces_p, alpha, beta)
                moments_p = body2wind(moments_p, alpha, beta)
                state.cancel_perturbation()

                # minus perturb
                eps_v0[i] = - eps/2
                eps_vec = wind2body(eps_v0, alpha, beta)
                state.perturbate(eps_vec, keyword)
                forces_m, moments_m = self.calculate_forces_and_moments(state, environment, controls)
                forces_m = body2wind(forces_m, alpha, beta)
                moments_m = body2wind(moments_m, alpha, beta)
                state.cancel_perturbation()

                k = names[keyword][i]
                for j in range(3):
                    # print(Fnames[j] + k, forces[j])
                    derivatives[Fnames[j] + k] = (forces_p[j] - forces_m[j]) / eps
                    derivatives[Mnames[j] + k] = (moments_p[j] - moments_m[j]) / eps

        return derivatives


class ConventionalControls:
    def __init__(self, controls_vec=None, N=None):
        self.info = 'controls.vec=[delta_aileron, delta_elevator,delta_rudder,delta_throttle]'
        if controls_vec is None:
            if N is None:
                self.vec = np.ones((1, 4))
            else:
                self.vec = np.ones((1, 4))
        else:
            # transpose if the controls are [n,m], we want it [m,n] so that controls.vec[1] is one contr example
            if controls_vec.ndim ==1:
                controls_vec = np.expand_dims(controls_vec, axis=0)
            self.vec = controls_vec
            self.N = self.vec.shape[0]

    def __repr__(self):
        rv = (
            "Conventional Controls \n"
            f"delta_aileron: {self.delta_aileron} \n"
            f"delta_elevator: {self.delta_elevator} \n"
            f"delta_rudder: {self.delta_rudder} \n"
            f"delta_throttle: {self.delta_throttle} \n"
        )
        return rv

    def evaluate_sequence(self, t, controls_sequence):
        sz = (1,4) if (type(t) == float or type(t) == np.float64 or len(t) == 1) \
            else (len(t), 4)
        self.vec = np.zeros(sz)
        for c_name, c_fun in controls_sequence.items():
            if hasattr(self, c_name):
                setattr(self, c_name, c_fun(t))
            else:
                raise AttributeError
        return self

    @property
    def delta_aileron(self):
        return self.vec.T[0]
    @delta_aileron.setter
    def delta_aileron(self, value):
        self.vec.T[0] = value

    @property
    def delta_elevator(self):
        return self.vec.T[1]
    @delta_elevator.setter
    def delta_elevator(self, value):
        self.vec.T[1] = value

    @property
    def delta_rudder(self):
        return self.vec.T[2]
    @delta_rudder.setter
    def delta_rudder(self, value):
        self.vec.T[2] = value

    @property
    def delta_throttle(self):
        return self.vec.T[3]
    @delta_throttle.setter
    def delta_throttle(self, value):
        self.vec.T[3] = value