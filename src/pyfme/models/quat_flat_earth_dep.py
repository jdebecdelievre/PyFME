# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.Distributed under the terms of the MIT License.

Quaternion Flat Earth 
---------------------

Classical aircraft motion equations assuming no Earth rotation
inertial effects, representing attitude quaternions and integrating aircraft position in Earth axis (Flat
Earth).
"""

import numpy as np
from numpy import sin, cos
import pdb
from pyfme.models.dynamic_system import AircraftDynamicSystem
from numba import jit
from pyfme.models.state import BodyAxisStateQuaternion
import quaternion as npquat
import pdb
from pyfme.utils.change_euler_quaternion import rotate_vector

QuatFlatEarthState = BodyAxisStateQuaternion


class QuatFlatEarth(AircraftDynamicSystem):
    """Euler Flat Earth Dynamic System.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with quaternions. Aircraft position is
    performed on Earth axis.
    """

    def __init__(self, aircraft = None, environment=None):
        super().__init__(aircraft, environment)
        self.info = ["x_e","y_e","z_e", "q0","qx","qy", "qz", "u","v","w", "p","q","r"]

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

        state_dot = self.make_state_obj()

        # get inertia parameters
        mass = self.aircraft.mass
        I = self.aircraft.inertia
        invI = self.aircraft.inertia_inverse

        # get state values
        u, v, w = state.velocity.T
        p, q, r = state.omega.T
        # phi, theta, psi = attitude.T

        # Linear momentum equations
        Fx, Fy, Fz = forces.T
        state_dot.u = Fx / mass + r * v - q * w
        state_dot.v = Fy / mass - r * u + p * w
        state_dot.w = Fz / mass + q * u - p * v

        # Angular momentum equations
        state_dot.p, state_dot.q, state_dot.r = (invI @ (moments - np.cross(state.omega, (I @ state.omega.T).T)).T)

        # Angular Kinematic equations
        Qomega = npquat.from_float_array(np.hstack((np.zeros((state.N,1)), state.omega)))
        state_dot.quaternion = 1/2 * state.quaternion * Qomega

        # Linear kinematic equations : rotate body velocity back to earth frame
        state_dot.position = rotate_vector(state.velocity, state.quaternion.conjugate())
        return state_dot.vec

    @jit
    def inverse_dynamics(self, time_step, 
                        position, quaternion,
                        earth_velocity=None, quaternion_dot=None):
        
        # Initialize state object
        N = quaternion.shape[0]
        state = EulerFlatEarthState(N=N)
        state.position = position
        state.quaternion = quaternion
        
        # Create state_dot
        state_dot = EulerFlatEarthState(N=N)
        if earth_velocity is None:
            state_dot.position = np.gradient(position, time_step, axis=0)
        else:
            state_dot.position = earth_velocity
        if quaternion_dot is None:
            state_dot.quaternion = np.gradient(quaternion, time_step, axis=0)
        else:
            state_dot.quaternion = quaternion_dot

        # Inverse kinematics
        state = self._inverse_kinematics(state, state_dot)

        # Differentiate body variables
        state_dot.velocity = np.gradient(state.velocity.values, time_step, axis=0)
        state_dot.omega = np.gradient(state.omega.values, time_step, axis=0)

        # Inverse momentum equations
        forces, moments = self._inverse_momentum_equations(state, state_dot)

        # Substract gravity (in place method)
        self._substract_gravity(forces, state)
        return forces, moments, state, state_dot


    # @jit
    # def _substract_gravity(self, forces, state, mass=None):
    #     mass = self.aircraft.mass if mass is None else mass
    #     conditions = self.environment.calculate_aero_conditions(state)
    #     forces -= conditions.gravity_vector*mass

    @jit
    def _inverse_kinematics(self, state, state_dot):
        """"
        Get rotation rates from attitude and body axis velocities from earth fixed ones
        """
        # Angular Kinematic equations
        state.omega = 2 * state_dot.quaternion * np.invert(state.quaternion)

        # Linear kinematic equations
        state.velocity = rotate_vector(state_dot.position, state.quaternion)

        return state

