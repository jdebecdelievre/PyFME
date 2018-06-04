# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
----------
Cessna 172
----------

References
----------
[1] ETKIN, Dynamics of Flight, Stability and Control
----------
"""


import numpy as np
import pdb

from pyfme.aircrafts.aircraft import Aircraft
from pyfme.models.state import AircraftState, EarthPosition, EulerAttitude, BodyVelocity
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.wind import NoWind
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.environment import Environment

class LinearB747(Aircraft):
    """
    Purely linear model of a Boeing 747 around a particular equilibrium condition
    """

    def __init__(self):

        # Mass & Inertia
        self.mass = 2.83176e6/9.81  # kg
        self.inertia = np.diag([.247e8, .449e8, .673e8])  # kg·m²
        self.inertia[0, 2] = -.212e7
        self.inertia[2, 0] = -.212e7

        # Geometry
        self.Sw = 511  # m2
        self.chord = 8.324  # m
        self.span = 59.64  # m

        # Aerodynamic Data# Values used for testing
        self.stability_derivatives = {
            'Xu': -1.982e3,
            'Xw': 4.025e3,
            'Xq': 0,
            'Xw_dot': 0,
            'Zu': -2.595e4,
            'Zw': -9.030e4,
            'Zq': -4.524e5,
            'Zw_dot': 1.909e3,
            'Mu': 1.593e4,
            'Mw': -1.563e5,
            'Mq': -1.521e7,
            'Mw_dot': -1.702e4,
            'Yv': -1.610e4,
            'Yp': 0,
            'Yr': 0,
            'Lv': -3.062e5,
            'Lp': -1.076e7,
            'Lr': 9.925e6,
            'Nv': 2.131e5,
            'Np': -1.330e6,
            'Nr': -8.934e6
        }

    def calculate_derivatives(self, state, environment, controls=None, eps=0):
        return self.stability_derivatives

    def trimmed_conditions(self):
        # state
        att = EulerAttitude(0, 0, 0) # from Etkin
        vel = BodyVelocity(235.9, 0, 0, att) # from Etkin
        pos = EarthPosition(0, 0, -1000) # arbitrary
        state =  AircraftState(pos, att, vel)

        # environment
        atmosphere = ISA1976()
        gravity = VerticalConstant()
        wind = NoWind()
        environment = Environment(atmosphere, gravity, wind)
        environment._rho = 0.3045

        return state, environment