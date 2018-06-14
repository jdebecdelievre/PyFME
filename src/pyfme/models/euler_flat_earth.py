# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Euler Flat Earth
----------------

Classical aircraft motion equations assuming no Earth rotation
inertial effects, representing attitude with Euler angles (not valid for
all-attitude flight) and integrating aircraft position in Earth axis (Flat
Earth).
"""

import numpy as np
from numpy import sin, cos
import pdb
from pyfme.models.dynamic_system import AircraftDynamicSystem
from numba import jit
from pyfme.models.dynamic_system import BodyAxisState
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


class EulerFlatEarth(AircraftDynamicSystem):
    """Euler Flat Earth Dynamic System.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with Euler angles. Aircraft position is
    performed on Earth axis.
    """

    def __init__(self, aircraft, environment):
        super().__init__(aircraft, environment)
        self.info = ["x_earth","y_earth","z_earth", "phi","theta","psi", "u","v","w", "p","q","r"]

    @jit
    def _system_equations(self, t, state_vec, controls_sequence):
        """Euler flat earth equations: linear momentum equations, angular momentum
        equations, angular kinematic equations, linear kinematic
        equations.

        Parameters
        ----------
        state_vec : [x,y,z, phi,theta,psi, u,v,w, p,q,r]

        Returns
        -------
        dstate_dt : array_like, shape(9)
            Derivative with respect to time of the state vector.
            Current value of absolute acceleration and angular acceleration,
            both expressed in body axes, Euler angles derivatives and velocity
            with respect to Earth Axis.
            (du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dtheta_dt, dphi_dt,
            dpsi_dt, dx_dt, dy_dt, dz_dt)
            (m/s² , m/s², m/s², rad/s², rad/s², rad/s², rad/s, rad/s, rad/s,
            m/s, m/s, m/s).

        References
        ----------
        .. [1] B. Etkin, Dynamics of flight, stability and control
        """
        state = BodyAxisState(state_vec)
        # get controls at time t
        controls = self.aircraft.get_controls(t, controls_sequence)

        # get forces and moments
        # Estimate aerodynamic conditions
        conditions = self.environment.calculate_aero_conditions(state)
        forces, moments = self.aircraft.calculate_forces_and_moments(state,
                                    conditions, controls)
        Fx, Fy, Fz = forces.T

        # get inertia parameters
        mass = self.aircraft.mass
        I = self.aircraft.inertia
        invI = self.aircraft.inertia_inverse

        # get state values
        u, v, w = state.body_vel
        omega = state.euler_ang_rate
        p, q, r = omega
        phi, theta, psi = state.euler_angles

        # Linear momentum equations
        du_dt = Fx / mass + r * v - q * w
        dv_dt = Fy / mass - r * u + p * w
        dw_dt = Fz / mass + q * u - p * v

        # Angular momentum equations
        dp_dt, dq_dt, dr_dt = invI @ (moments - np.cross(omega.T, (I @ omega).T)).T

        # Precompute sines and cosines
        cos_phi = cos(phi)
        cos_theta = cos(theta)
        cos_psi = cos(psi)
        sin_phi = sin(phi)
        sin_theta = sin(theta)
        sin_psi = sin(psi)

        # Angular Kinematic equations (nonINFcheck to prevent blow up)
        dtheta_dt = q * cos_phi - r * sin_phi
        dphi_dt = p + (q * sin_phi + r * cos_phi) * nonINFchecked(np.tan(theta))
        dpsi_dt = (q * sin_phi + r * cos_phi) * nonINFchecked(1/cos_theta)

        # Linear kinematic equations
        dx_dt = (cos_theta * cos_psi * u +
                 (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * v +
                 (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * w)
        dy_dt = (cos_theta * sin_psi * u +
                 (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * v +
                 (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * w)
        dz_dt = -u * sin_theta + v * sin_phi * cos_theta + w * cos(
            phi) * cos_theta

        return np.vstack((dx_dt, dy_dt, dz_dt, dphi_dt, dtheta_dt, dpsi_dt,
                         du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt)).T

# Define State for Euler Flat Earth
EulerFlatEarthState = BodyAxisState

def nonINFchecked(array, M=1e5):
    b = np.abs(array) < M
    return array*b + M*(1-b)
