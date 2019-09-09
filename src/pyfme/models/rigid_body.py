# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.Distributed under the terms of the MIT License.


Rigid body dynamic systems with flat earth assumption. Euler angles or quaternions description of attitudes available
"""

import numpy as np
from numpy import sin, cos
from pyfme.models.dynamic_system import AircraftDynamicSystem
from numba import jit
from pyfme.models.state import BodyAxisState, BodyAxisStateQuaternion
import quaternion as npquat
from pyfme.utils.change_euler_quaternion import change_basis
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0
import pdb

class RigidBodySystem(AircraftDynamicSystem):
    # @jit
    def _substract_gravity(self, forces, state, mass=None):
        mass = self.aircraft.mass if mass is None else mass
        conditions = self.environment.calculate_aero_conditions(state)
        forces -= conditions.gravity_vector*mass
        return forces

    # @jit
    def inverse_momentum_equations(self, state, state_dot,
                                   mass=None,inertia_matrix=None):
        """flat earth equations: linear momentum equations, angular momentum
        equations used to get forces from body rates and velocitities + derivatives
        """
        # Get inertia parameters
        assert (mass is not None or self.aircraft is not None),"mass must be specified if self.aircraft is None"
        assert (inertia_matrix is not None or self.aircraft is not None),"inertia_matrix must be specified if self.aircraft is None"
        mass = mass if mass != None else self.aircraft.mass
        I = inertia_matrix if inertia_matrix != None else self.aircraft.inertia

        # Linear momentum equations
        forces = mass*state_dot.velocity + np.cross(state.omega, mass*state.velocity)

        # Angular momentum equations
        moments = (I@ state_dot.omega.T).T + np.cross(state.omega, (I@state.omega.T).T)
        return forces, moments

    # @jit
    def inverse_dynamics(self, time_step, 
                        position, attitude,
                        earth_velocity=None, attitude_dot=None):
        '''
        Inverse function : from position and attitude in Earth frame, get forces and moments
        
        Parameters
        ----------
        time_step : delta_t for differenciation 
        position, attitude, earth_velocity, attitude_dot : all nd arrays of size (N,12 or 13)
        Attitude can be euler angles (phi theta psi) in rad or quaternions (q0, qx, qy, qz)

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
        '''

        # figure out if attitudes are euler angles or quaternions
        if attitude.shape[1] == 3:
            quat = False
        elif attitude.shape[1] == 4:
            quat = True
        else:
            raise AttributeError('Wrong size of attitude vector')

        # Initialize state object
        N = attitude.shape[0]
        state = self.make_state_obj(N=N)
        state.position = position
        if quat:
            state.quaternion = attitude
        else:
            state.attitude = attitude
        
        # Create state_dot
        state_dot = self.make_state_obj(N=N)
        if earth_velocity is None:
            state_dot.position = np.gradient(position, time_step, axis=0, edge_order=2)
        else:
            state_dot.position = earth_velocity
        if attitude_dot is None:
            if quat:
                state_dot.quaternion = np.gradient(attitude, time_step, axis=0, edge_order=2)
            else:
                state_dot.attitude = np.gradient(attitude, time_step, axis=0, edge_order=2)
        else:
            if quat:
                state_dot.quaternion = attitude_dot
            else:
                state_dot.attitude = attitude_dot

        # Inverse kinematics
        state = self._inverse_kinematics(state, state_dot)

        # Differentiate body variables
        state_dot.velocity = np.gradient(state.velocity, time_step, axis=0, edge_order=2)
        state_dot.omega = np.gradient(state.omega, time_step, axis=0, edge_order=2)

        # Inverse momentum equations
        forces, moments = self.inverse_momentum_equations(state, state_dot)

        # Substract gravity (in place method)
        self._substract_gravity(forces, state)

        return forces, moments, state, state_dot



    
'''
Euler Flat Earth
----------------

Classical aircraft motion equations assuming no Earth rotation
inertial effects, representing attitude with Euler angles (not valid for
all-attitude flight) and integrating aircraft position in Earth axis (Flat
Earth).
'''

# Define State for Euler Flat Earth
RigidBodyEulerState = BodyAxisState

class RigidBodyEuler(RigidBodySystem):
    """Euler Flat Earth Dynamic System.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with Euler angles. Aircraft position is
    performed on Earth axis.
    """

    def __init__(self, aircraft = None, environment=None):
        super().__init__(aircraft, environment)
        self.info = ["x_e","y_e","z_e", "phi","theta","psi", "u","v","w", "p","q","r"]

    def make_state_obj(self, state_vec=None, **options):
        return RigidBodyEulerState(state_vec, **options)

    # @jit
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
    def _inverse_kinematics(self, state, state_dot):
        """"
        Get Roll rates from attitude and body axis velocities from earth fixed ones
        """

        phi, theta, psi = state.attitude.T
        phi_dot, theta_dot, psi_dot = state_dot.attitude.T
        xdot, ydot, zdot = state_dot.position.T

        # Precompute sines and cosines
        cos_phi = cos(phi)
        cos_theta = cos(theta)
        cos_psi = cos(psi)
        sin_phi = sin(phi)
        sin_theta = sin(theta)
        sin_psi = sin(psi)

        # Angular Kinematic equations
        state.p = phi_dot - sin_theta * psi_dot
        state.q = cos_phi * theta_dot + sin_phi * cos_theta * psi_dot
        state.r = -sin_phi * theta_dot + cos_phi * cos_theta * psi_dot

        # Linear kinematic equations
        """
        In wolfram:
        {{cos(psi), -sin(psi),0}, {sin(psi), cos(psi),0},{0,0,1}}
        *{{cos(theta), 0,sin(theta)}, {0,1,0},{-sin(theta), 0,cos(theta)}}
        *{{1,0,0},{0, cos(phi), -sin(phi)}, {0,sin(phi), cos(phi)}}
        """
        state.u = cos_theta * cos_psi * xdot +\
            cos_theta * sin_psi * ydot +\
            - sin_theta * zdot
        state.v = (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * xdot +\
            (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * ydot +\
            (sin_phi * cos_theta) * zdot
        state.w = (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * xdot +\
            (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * ydot +\
            (cos_phi * cos_theta) * zdot
        
        return state

'''
Quaternion Flat Earth 
---------------------

Classical aircraft motion equations assuming no Earth rotation
inertial effects, representing attitude quaternions and integrating aircraft position in Earth axis (Flat
Earth).
'''

# Define adequate state
RigidBodyQuatState = BodyAxisStateQuaternion

class RigidBodyQuat(RigidBodySystem):
    """Flat Earth Dynamic System with quaternions.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with quaternions. Aircraft position is
    performed on Earth axis.
    """

    def __init__(self, aircraft = None, environment=None):
        super().__init__(aircraft, environment)
        self.info = ["x_e","y_e","z_e", "q0","qx","qy", "qz", "u","v","w", "p","q","r"]

    def make_state_obj(self, state_vec=None, **options):
        return RigidBodyQuatState(state_vec, **options)

    # @jit
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

        state_dot = self.make_state_obj(N=state.N)

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
        state_dot.quaternion = 1/2 * state.quaternion * Qomega + 1/2*(1-np.norm(state.quaternion))*state.quaternion/.1

        # Linear kinematic equations : rotate body velocity back to earth frame
        state_dot.position = change_basis(state.velocity, state.quaternion.conjugate())

        self.aircraft.state_dot = state_dot
        return state_dot.vec


    @jit
    def _inverse_kinematics(self, state, state_dot):
        """"
        Get rotation rates from attitude and body axis velocities from earth fixed ones
        """
        # Angular Kinematic equations
        # state.omega = 2 * npquat.as_float_array(state_dot.quaternion * np.conjugate(state.quaternion))[:,1:]
        state.omega = npquat.as_float_array(2 * state.quaternion.conjugate() * state_dot.quaternion)[:,1:]

        # Linear kinematic equations
        state.velocity = change_basis(state_dot.position, state.quaternion)

        return state


def nonINFchecked(array, M=1e5):
    b = np.abs(array) < M
    return array*b + M*(1-b)