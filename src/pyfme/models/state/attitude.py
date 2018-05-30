"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Attitude
--------
Aircraft attitude representations prepared to store the aircraft orientation in
Euler angles and quaternions independently of the dynamic system used.
"""
from abc import abstractmethod

import numpy as np


class Attitude:
    """Attitude

    Attributes
    ----------

    euler_angles : ndarray, shape(3)
        (theta [rad], phi [rad], psi [rad])
    theta
    phi
    psi
    quaternions : ndarray, shape(4)
        (q0, q1, q2, q3)
    q0
    q1
    q2
    q3
    """

    def __init__(self):
        # Euler angles (psi, theta, phi)
        self._euler_angles = np.zeros(3)  # rad
        # Quaternions (q0, q1, q2, q3)
        self._quaternions = np.zeros(4)
        self._euler_angles_ref = None

    @abstractmethod
    def update(self, value):
        raise NotImplementedError

    @property
    def euler_angles(self):
        return self._euler_angles

    @property
    def psi(self):
        return self._euler_angles[2]

    @property
    def theta(self):
        return self._euler_angles[0]

    @property
    def phi(self):
        return self._euler_angles[1]

    @property
    def quaternions(self):
        return self._quaternions

    @property
    def q0(self):
        return self._quaternions[0]

    @property
    def q1(self):
        return self._quaternions[1]

    @property
    def q2(self):
        return self._quaternions[2]

    @property
    def q3(self):
        return self._quaternions[3]

    @property
    def value(self):
        """Only for testing purposes"""
        return np.hstack((self.euler_angles, self.quaternions))


class EulerAttitude(Attitude):

    def __init__(self, theta, phi, psi):
        # TODO: docstring
        super().__init__()
        self.update(np.array([theta, phi, psi]))

    def update(self, value):
        self._euler_angles[:] = value
        # TODO: transform quaternions to Euler angles
        self._quaternions = np.zeros(4)

    def perturbate(self, eps_vector, **kwargs):
        assert self._euler_angles_ref is None, "Cancel perturbation on velocity before perturbating again"
        self._euler_angles_ref = np.copy(self._euler_angles)
        self.update(self._euler_angles + eps_vector)

    def cancel_perturbation(self, **kwargs):
        if self._euler_angles_ref is not None:
            self.update(self._euler_angles_ref)
            self._euler_angles_ref = None

    def __repr__(self):
        rv = (f"theta: {self.theta:.3f} rad, phi: {self.phi:.3f} rad, "
              f"psi: {self.psi:.3f} rad")
        return rv


class QuaternionAttitude(Attitude):
    def __init__(self, q0, q1, q2, q3):
        # TODO: docstring
        super().__init__()
        self.update(np.array([q0, q1, q2, q3]))

    def update(self, value):
        self._quaternions[:] = value
        # TODO: transform quaternions to Euler angles
        self._euler_angles = np.zeros(3)
