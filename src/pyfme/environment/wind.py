"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Wind Models
-----------

"""
import numpy as np


class NoWind(object):

    def horizon(self, state):
        # Wind velocity: FROM North to South, FROM East to West,
        return np.zeros((state.N, 3), dtype=float)

    def body(self, state):
        # Wind velocity in the UPSIDE direction
        return np.zeros((state.N, 3), dtype=float)