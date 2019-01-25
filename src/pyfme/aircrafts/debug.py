# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
----------
simple possible model for debug
----------

"""
import numpy as np
import pdb
import json

from pyfme.aircrafts.aircraft import Aircraft, ConventionalControls
from pyfme.models.constants import slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body, body2wind
from copy import deepcopy as cp
from collections import namedtuple
from pyfme.environment import Conditions
from pyfme.aircrafts import BasisLinear

class Debug(Aircraft):

    def __init__(self):

        super().__init__()
        self.mass = .5
        # self.inertia = np.eye(3)*4
        self.inertia = np.array([[1,2,3],[2,4,5],[3,5,6]])
        self.inertia_inverse = np.linalg.inv(self.inertia)

    def _calculate_aero_forces_moments(self, conditions, state, controls):
        D = 0.1
        Y = 0.1
        L = 1
        l = 1/10
        m = 1/10
        n =1/10

        return L, D, Y, l, m, n

    def _calculate_thrust_forces_moments(self, TAS, conditions, controls):
        return 0

    def get_controls(self, t, controls_sequence):
        return ConventionalControls().evaluate_sequence(t, controls_sequence)
