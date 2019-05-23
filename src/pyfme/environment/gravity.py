"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Gravity Models
--------------

"""
import numpy as np

from pyfme.models.constants import GRAVITY, STD_GRAVITATIONAL_PARAMETER
from pyfme.utils.coordinates import hor2body
from pyfme.utils.change_euler_quaternion import change_basis

class VerticalConstant(object):
    """Vertical constant gravity model.
    """

    def __init__(self, magnitude=GRAVITY):
        self._magnitude = magnitude

    def vector(self, state):
        _versor = grav_versor(state)
        return self._magnitude * _versor


class VerticalNewton(object):
    """Vertical gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        pass

    def vector(self, state):
        r_squared = (state.coord_geocentric @
                     state.coord_geocentric)
        _magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        _versor = grav_versor(state)
        return _magnitude * _versor


class LatitudeModel(object):
    # TODO: https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude_model

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def vector(self, system):
        raise NotImplementedError

def grav_versor(state):
    if hasattr(state, 'quaternion'):
        _versor = change_basis(np.array([[0,0,1]]*state.N), state.quaternion)
    else:
        _versor = np.array([- np.sin(state.theta),
                            np.sin(state.phi)*np.cos(state.theta),
                            np.cos(state.phi)*np.cos(state.theta)]).T
    return _versor    