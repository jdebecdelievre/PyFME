"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""
from pyfme.utils.anemometry import tas2eas, tas2cas, calculate_alpha_beta_TAS
from collections import namedtuple
# Conditions class
Conditions = namedtuple('conditions', ['TAS','CAS','Mach','q_inf',
                                       'rho','T','P','a','alpha',
                                       'beta','gravity_vector'])

from pyfme.environment.atmosphere import SeaLevel
from pyfme.environment.wind import NoWind
from pyfme.environment.gravity import VerticalConstant
import numpy as np

class Environment(object):
    """
    Stores all the environment info: atmosphere, gravity and wind.
    """

    def __init__(self, atmosphere=None, gravity=None, wind=None):
        """
        Parameters
        ----------
        atmosphere : Atmosphere
            Atmospheric model.
        gravity : Gravity
            Gravity model.
        wind : Wind
            Wind or gust model.
        """
        self.atmosphere = atmosphere if atmosphere else SeaLevel()
        self.gravity = gravity if gravity else VerticalConstant()
        self.wind = wind if wind else NoWind()

    def gravity_magnitude(self, state):
        return self.gravity.magnitude(state)

    def gravity_vector(self, state):
        return self.gravity.vector(state)

    def horizon_wind(self, state):
        return self.wind.horizon(state)

    def body_wind(self, state):
        return self.wind.body(state)

    def calculate_aero_conditions(self, state):

        # Getting conditions from environment
        body_wind = self.body_wind(state)
        T, P, rho, a = self.atmosphere.variables(state)

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = (state.velocity - body_wind)
        alpha, beta, TAS = calculate_alpha_beta_TAS(aero_vel)

        # Setting velocities & dynamic pressure
        CAS = tas2cas(TAS, P, rho)
        EAS = tas2eas(TAS, rho)
        Mach = TAS / a
        q_inf = 0.5 * rho * np.square(TAS)

        # gravity vector
        gravity_vector = self.gravity_vector(state)
        return Conditions(TAS, CAS, Mach, q_inf, rho, T, P, a, alpha, beta, gravity_vector)
