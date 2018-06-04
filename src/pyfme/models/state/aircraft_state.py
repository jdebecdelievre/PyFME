# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Aircraft State
--------------

"""

import numpy as np

from .angular_velocity import BodyAngularVelocity
from .acceleration import BodyAcceleration
from .angular_acceleration import BodyAngularAcceleration
from pyfme.utils.coordinates import wind2body
from json import dump


class AircraftState:
    def __init__(self, position, attitude, velocity, angular_vel=None,
                 acceleration=None, angular_accel=None):

        self.position = position
        self.attitude = attitude
        self.velocity = velocity

        if angular_vel is None:
            angular_vel = BodyAngularVelocity(0, 0, 0, attitude)
        if acceleration is None:
            acceleration = BodyAcceleration(0, 0, 0, attitude)
        if angular_accel is None:
            angular_accel = BodyAngularAcceleration(0, 0, 0, attitude)

        self.angular_vel = angular_vel
        self.acceleration = acceleration
        self.angular_accel = angular_accel


    @property
    def value(self):
        """Only for testing purposes"""
        return np.hstack((self.position.value, self.attitude.value,
                          self.velocity.value, self.angular_vel.value,
                          self.acceleration.value, self.angular_accel.value))

    def __repr__(self):
        rv = (
            "Aircraft State \n"
            f"{self.position} \n"
            f"{self.attitude} \n"
            f"{self.velocity} \n"
            f"{self.angular_vel} \n"
            f"{self.acceleration} \n"
            f"{self.angular_accel} \n"
        )
        return rv

    def perturbate(self, eps_vector, keyword):
        """
        Perturbates the "keyword" part of the state (position, attitude, velocity, angular_vel) by eps_vec (size (3,)).
        Each vector V becomes V + eps_vector, so eps is the change in each direction.
        The perturbations can optionally be specified in the stability_axis. Note that it is common to linearize the
        dynamics in stability axis
        """
        # Get the "keyword" part of the state
        attr = getattr(self, keyword)

        # Perturbate
        attr.perturbate(eps_vector, attitude=self.attitude)

    def cancel_perturbation(self):
        """
        Brings back to reference state.
        """
        for keyword in ['position', 'attitude', 'velocity', 'angular_vel', 'acceleration']:
            getattr(self, keyword).cancel_perturbation(attitude=self.attitude)

        return self

    def save_to_json(self, filename):
        state = dict()

        state['x_e'] = self.position.x_earth
        state['y_e'] = self.position.y_earth
        state['z_e'] = self.position.z_earth

        state['phi'] = self.attitude.phi
        state['theta'] = self.attitude.theta
        state['psi'] = self.attitude.psi

        state['u'] = self.velocity.u
        state['v'] = self.velocity.v
        state['w'] = self.velocity.w

        state['p'] = self.angular_vel.p
        state['q'] = self.angular_vel.q
        state['r'] = self.angular_vel.r

        with open(filename, 'w') as f:
            dump(state, f)