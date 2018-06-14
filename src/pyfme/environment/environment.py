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

class Environment(object):
    """
    Stores all the environment info: atmosphere, gravity and wind.
    """

    def __init__(self, atmosphere, gravity, wind):
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
        self.atmosphere = atmosphere
        self.gravity = gravity
        self.wind = wind

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
        rho, T, P, a = self.atmosphere.variables(state)

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = state.body_vel - body_wind
        alpha, beta, TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2]
        )

        # Setting velocities & dynamic pressure
        CAS = tas2cas(TAS, P, rho)
        EAS = tas2eas(TAS, rho)
        Mach = TAS / a
        q_inf = 0.5 * rho * TAS ** 2

        # gravity vector
        gravity_vector = self.gravity_vector(state)
        return Conditions(TAS, CAS, Mach, q_inf, rho, T, P, a, alpha, beta, gravity_vector)
