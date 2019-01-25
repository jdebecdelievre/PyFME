# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.Distributed under the terms of the MIT License.

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
from pyfme.models.state import BodyAxisState
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


class EulerFlatEarth(AircraftDynamicSystem):
    """Euler Flat Earth Dynamic System.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with Euler angles. Aircraft position is
    performed on Earth axis.
    """

    def __init__(self, aircraft = None, environment=None):
        super().__init__(aircraft, environment)
        self.info = ["x_e","y_e","z_e", "phi","theta","psi", "u","v","w", "p","q","r"]

    def make_state_obj(self, state_vec=None, **options):
        return QuatFlatEarthState(state_vec, **options)

    @jit
    def _system_equations(self, state, forces, moments):
        """Euler flat earth equations: linear momentum equations, angular momentum
        equations, angular kinematic equations, linear kinematic
        equations.

        Parameters
        ----------
        velocity, omega, attitude, forces, moments : all nd arrays of size (m,12)

        Returns
        -------
        dstate_dt : array_like, shape(9)
            Derivative with respect to time of the state vector.
            Current value of absolute acceleration and angular acceleration,
            both expressed in body axes, Euler angles derivatives and velocity
            with respect to Earth Axis.
            (du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dphi_dt, dtheta_dt,
            dpsi_dt, dx_dt, dy_dt, dz_dt)
            (m/s² , m/s², m/s², rad/s², rad/s², rad/s², rad/s, rad/s, rad/s,
            m/s, m/s, m/s).

        References
        ----------
        .. [1] B. Etkin, Dynamics of flight, stability and control
        """

        # get inertia parameters
        mass = self.aircraft.mass
        I = self.aircraft.inertia
        invI = self.aircraft.inertia_inverse

        # get state values
        phi, theta, psi = state.attitude.T
        u, v, w = state.velocity.T
        p, q, r = state.omega.T   
        Fx, Fy, Fz = forces.T

        # Translational dynamics
        du_dt = Fx / mass + r * v - q * w
        dv_dt = Fy / mass - r * u + p * w
        dw_dt = Fz / mass + q * u - p * v

        # Angular momentum equations
        dp_dt, dq_dt, dr_dt = (invI @ (moments - np.cross(state.omega, (I @ state.omega.T).T)).T)

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
        dz_dt = -u * sin_theta + v * sin_phi * cos_theta + w * cos_phi * cos_theta

        # return np.vstack((du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt)).T
        return np.vstack((dx_dt, dy_dt, dz_dt, dphi_dt, dtheta_dt, dpsi_dt,
                         du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt)).T

    @jit
    def inverse_dynamics(self, time_step, position, attitude, earth_velocity, attitude_dot,
                         earth_acceleration=None, attitude_dbldot=None):
        # Initialize state object
        N = attitude.shape[0]
        state = EulerFlatEarthState(N=N)
        state.position = position
        state.attitude = attitude
        state_dot = EulerFlatEarthState(N=N)
        state_dot.position = earth_velocity
        state_dot.attitude = attitude_dot

        # Inverse kinematics
        state.velocity, state.omega = self._inverse_kinematics(attitude, attitude_dot, earth_velocity)

        # Differentiate body velocities and rotation rates
        if earth_acceleration is None or attitude_dbldot is None:
            state_dot.velocity = np.gradient(state.velocity, time_step, axis=0)
            state_dot.omega = np.gradient(state.omega, time_step, axis=0)
        else:
            state_dot.velocity, state_dot.omega = self._inverse_kinematics_dot(attitude_dot, attitude_dbldot,
                                                                               earth_velocity, earth_acceleration)

        # Inverse momentum equations
        forces, moments = self._inverse_momentum_equations(state.velocity, state.omega, state_dot.velocity, state_dot.omega)

        # Substract gravity (in place method)
        self._substract_gravity(forces, state)
        return forces, moments, state, state_dot

    @jit
    def _inverse_momentum_equations(self, velocity, omega, velocity_dot, omega_dot,
                                   mass=None,inertia_matrix=None):
        """Euler flat earth equations: linear momentum equations, angular momentum
        equations used to get forces from body rates and velocitities + derivatives
        """
        # Get inertia parameters
        assert (mass is not None or self.aircraft is not None),"mass must be specified if self.aircraft is None"
        assert (inertia_matrix is not None or self.aircraft is not None),"inertia_matrix must be specified if self.aircraft is None"
        mass = mass if mass != None else self.aircraft.mass
        I = inertia_matrix if inertia_matrix != None else self.aircraft.inertia

        # Linear momentum equations
        forces = mass*velocity_dot + np.cross(omega, mass*velocity)

        # Angular momentum equations
        moments = (I@ omega_dot.T).T + np.cross(omega, (I@omega.T).T)
        return forces, moments

    @jit
    def _substract_gravity(self, forces, state, mass=None):
        mass = self.aircraft.mass if mass is None else mass
        conditions = self.environment.calculate_aero_conditions(state)
        forces -= conditions.gravity_vector*mass

    @jit
    def _inverse_kinematics(self, attitude, attitude_dot, earth_velocities):
        """"
        Get Roll rates from attitude and body axis velocities from earth fixed ones
        """
        phi, theta, psi = attitude.T
        phi_dot, theta_dot, psi_dot = attitude_dot.T
        xdot, ydot, zdot = earth_velocities.T

        # Precompute sines and cosines
        cos_phi = cos(phi)
        cos_theta = cos(theta)
        cos_psi = cos(psi)
        sin_phi = sin(phi)
        sin_theta = sin(theta)
        sin_psi = sin(psi)

        # Angular Kinematic equations
        p = phi_dot - sin_theta * psi_dot
        q = cos_phi*theta_dot + sin_phi * cos_theta * psi_dot
        r = -sin_phi*theta_dot + cos_phi * cos_theta * psi_dot

        # Linear kinematic equations
        """
        In wolfram:
        {{cos(psi), -sin(psi),0}, {sin(psi), cos(psi),0},{0,0,1}}
        *{{cos(theta), 0,sin(theta)}, {0,1,0},{-sin(theta), 0,cos(theta)}}
        *{{1,0,0},{0, cos(phi), -sin(phi)}, {0,sin(phi), cos(phi)}}
        """
        u = cos_theta * cos_psi * xdot +\
            cos_theta * sin_psi * ydot +\
            - sin_theta * zdot
        v = (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * xdot +\
            (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * ydot +\
            (sin_phi * cos_theta) * zdot
        w = (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * xdot +\
            (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * ydot +\
            (cos_phi * cos_theta) * zdot
        return np.array([u, v, w]).T, np.array([p, q, r]).T



    def _inverse_kinematics_dot(self, attitude_dot, attitude_dbldot, earth_velocity, earth_acceleration):
        raise NotImplementedError

# Define State for Euler Flat Earth
EulerFlatEarthState = BodyAxisState

def nonINFchecked(array, M=1e5):
    b = np.abs(array) < M
    return array*b + M*(1-b)

