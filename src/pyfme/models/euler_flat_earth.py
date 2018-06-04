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
from copy import deepcopy as dcp
import pdb
from pyfme.models.dynamic_system import AircraftDynamicSystem
from pyfme.models.state import (
    AircraftState, EarthPosition, EulerAttitude, BodyVelocity,
    BodyAngularVelocity, BodyAcceleration, BodyAngularAcceleration
)
from pyfme.utils.coordinates import body2wind, wind2body
import math

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

class EulerFlatEarth(AircraftDynamicSystem):
    """Euler Flat Earth Dynamic System.

    Classical aircraft motion equations assuming no Earth rotation, no Earth
    curvature and modelling attitude with Euler angles. Aircraft position is
    performed on Earth axis.
    """

    def fun(self, t, x=None):

        if x is not None:
            # update full state if necessary
            self._update_full_system_state_from_state(x, self.state_vector_dot)

        updated_simulation = self.update_simulation(t, self.full_state)

        mass = updated_simulation.aircraft.mass
        inertia = updated_simulation.aircraft.inertia
        forces = updated_simulation.aircraft.total_forces
        moments = updated_simulation.aircraft.total_moments

        rv = _system_equations(t, x, mass, inertia, forces, moments)

        return rv

    def right_hand_side(self, full_state, environment, aircraft,
                              controls):
        try:
            environment.update(full_state)
        except:
            pdb.set_trace()
        aircraft.calculate_forces_and_moments(full_state, environment,
                                              controls)

        mass = aircraft.mass
        inertia = aircraft.inertia
        forces = aircraft.total_forces
        moments = aircraft.total_moments

        t0 = 0
        x0 = self._get_state_vector_from_full_state(full_state)

        return _system_equations(t0, x0, mass, inertia, forces, moments)

    def steady_state_trim_fun(self, full_state, environment, aircraft,
                              controls):
        return self.right_hand_side(full_state, environment, aircraft, controls)[:6]

    def _update_full_system_state_from_state(self, state, state_dot):

        self.full_state.position.update(state[9:12])
        self.full_state.attitude.update(state[6:9])
        att = self.full_state.attitude
        self.full_state.velocity.update(state[0:3], att)
        self.full_state.angular_vel.update(state[3:6], att)

        self.full_state.acceleration.update(state_dot[0:3], att)
        self.full_state.angular_accel.update(state_dot[3:6], att)

    def _adapt_full_state_to_dynamic_system(self, full_state):

        pos = EarthPosition(full_state.position.x_earth,
                            full_state.position.y_earth,
                            full_state.position.height,
                            full_state.position.lat,
                            full_state.position.lon)

        att = EulerAttitude(full_state.attitude.phi,
                            full_state.attitude.theta,
                            full_state.attitude.psi)

        vel = BodyVelocity(full_state.velocity.u,
                           full_state.velocity.v,
                           full_state.velocity.w,
                           att)

        ang_vel = BodyAngularVelocity(full_state.angular_vel.p,
                                      full_state.angular_vel.q,
                                      full_state.angular_vel.r,
                                      att)

        accel = BodyAcceleration(full_state.acceleration.u_dot,
                                 full_state.acceleration.v_dot,
                                 full_state.acceleration.w_dot,
                                 att)

        ang_accel = BodyAngularAcceleration(full_state.angular_accel.p_dot,
                                            full_state.angular_accel.q_dot,
                                            full_state.angular_accel.r_dot,
                                            att)

        full_state = AircraftState(pos, att, vel, ang_vel, accel, ang_accel)
        return full_state

    def _get_state_vector_from_full_state(self, full_state):

        x0 = np.array(
            [
                full_state.velocity.u,
                full_state.velocity.v,
                full_state.velocity.w,
                full_state.angular_vel.p,
                full_state.angular_vel.q,
                full_state.angular_vel.r,
                full_state.attitude.theta,
                full_state.attitude.phi,
                full_state.attitude.psi,
                full_state.position.x_earth,
                full_state.position.y_earth,
                full_state.position.z_earth
            ]
        )
        return x0

    def linearized_model(self, state, aircraft, environment, controls=None, method="direct", eps=1e-3):
        """
        Outputs matrices A_long and A_lat that are the lateral and longitudinal state matrices for the linearized system.
        As done in Etkin [2], these matrices are useful in stability axis.
        method can be:
            - "direct", in which case we compute the derivative of the accelerations : X_dot = f(X,U), so
                Aij = dfi/dxj(X,U)
            - "from forces", in which case we compute the dimensional force derivatives and use formulas in Etkin
            (/!\ contains assumptions on the point at which we linearize
        """

        if method=="from_forces":
            # get derivatives
            d = aircraft.calculate_derivatives(state, environment, controls,eps)

            # recover state variables
            u, v, w = state.velocity.vel_body
            alpha = np.arctan2(w,u)
            beta = np.arcsin(v/np.sqrt(u**2 + v**2 + w**2))
            theta = np.copy(state.attitude.theta) - alpha
            u, v, w = body2wind(state.velocity.vel_body, alpha, beta)
            g = environment.gravity_magnitude

            # get inertias (move them to stability axis)
            m = aircraft.mass
            Lwb = np.array([[cos(alpha) * cos(beta),sin(beta),sin(alpha) * cos(beta)],
                            [- cos(alpha) * sin(beta),cos(beta),-sin(alpha) * sin(beta)],
                            [-sin(alpha), 0, cos(alpha)]])
            I = (Lwb.dot(aircraft.inertia)).dot(Lwb.T)
            Ix = I[0,0]
            Iy = I[1, 1]
            Iz = I[2, 2]
            Ixz = - I[0, 2]
            Ixprime = (Ix*Iz - Ixz**2)/Iz
            Izprime = (Ix*Iz - Ixz**2)/Ix
            Ixzprime = Ixz/(Ix*Iz - Ixz**2)

            # Longitudinal matrix
            # Todo : add alpha_dot derivatives
            A1 = np.array([d['Xu'] / m, d['Xw'] / m, 0, -g*np.cos(theta)])
            A2 = np.array([d['Zu'], d['Zw'], d['Zq'] + m*u, -m*g*np.sin(theta)])/(m - d['Zw_dot'])
            A3 = (np.array([d['Mu'], d['Mw'], d['Mq'], 0]) + A2*d['Mw_dot']) / Iy
            A4 = np.array([0, 0, 1, 0])
            A_long = np.vstack((A1, A2, A3, A4))

            # Lateral dynamics
            A1 = np.array([d['Yv']/m, d['Yp']/m, d['Yr']/m - u, g*np.cos(theta)])
            A2 = np.array([d['Lv']/Ixprime + d['Nv']*Ixzprime, d['Lp']/Ixprime + d['Np']*Ixzprime,
                           d['Lr']/Ixprime + d['Nr']*Ixzprime, 0])
            A3 = np.array([d['Lv']*Ixzprime + d['Nv']/Izprime, d['Lp']*Ixzprime + d['Np']/Izprime,
                           d['Lr'] * Ixzprime + d['Nr'] / Izprime, 0])
            A4 = np.array([0, 1, np.tan(theta), 0])
            A_lat = np.vstack((A1, A2, A3, A4))

        elif method=="direct":
            # Rotation for stability derivatives in stability axis
            V = np.sqrt(state.velocity.u ** 2 + state.velocity.v ** 2 + state.velocity.w ** 2)
            alpha = np.arctan2(state.velocity.w, state.velocity.u)
            beta = np.arcsin(state.velocity.v / V)

            derivatives = {}
            for keyword in ['velocity', 'angular_vel', 'attitude']:
                derivatives[keyword] = {}
                for i in range(3):
                    derivatives[keyword][i] = {}
                    eps_v0 = np.zeros(3)

                    # plus perturb
                    eps_v0[i] = eps / 2
                    if keyword == 'attitude':
                        eps_vec = eps_v0
                    else:
                        eps_vec = wind2body(eps_v0, alpha, beta)
                    state.perturbate(eps_vec, keyword)
                    state_dot = self.right_hand_side(state, environment, aircraft, controls)
                    accel_p = body2wind(state_dot[0:3], alpha, beta)
                    ang_accel_p = body2wind(state_dot[3:6], alpha, beta)
                    angle_der_p = state_dot[6:9]
                    state.cancel_perturbation()

                    # minus perturb
                    eps_v0[i] = - eps / 2
                    if keyword == 'attitude':
                        eps_vec = eps_v0
                    else:
                        eps_vec = wind2body(eps_v0, alpha, beta)
                    state.perturbate(eps_vec, keyword)
                    state_dot = self.right_hand_side(state, environment, aircraft, controls)
                    accel_m = body2wind(state_dot[0:3], alpha, beta)
                    ang_accel_m = body2wind(state_dot[3:6], alpha, beta)
                    angle_der_m = state_dot[6:9]
                    state.cancel_perturbation()

                    derivatives[keyword][i]["acceleration"] = (accel_p - accel_m)/eps
                    derivatives[keyword][i]["angular_accel"] = (ang_accel_p - ang_accel_m)/eps
                    derivatives[keyword][i]["angle_der"] = (angle_der_p - angle_der_m)/eps

            # Longitudinal
            # line1 : d(delta_u_dot)/dall
            A1 = np.array([derivatives['velocity'][0]["acceleration"][0], derivatives['velocity'][2]["acceleration"][0],
                           derivatives['angular_vel'][1]["acceleration"][0], derivatives['attitude'][0]["acceleration"][0]
            ])
            # line2 : d(w_dot)/dall
            A2 = np.array([derivatives['velocity'][0]["acceleration"][2], derivatives['velocity'][2]["acceleration"][2],
                           derivatives['angular_vel'][1]["acceleration"][2], derivatives['attitude'][0]["acceleration"][2]
            ])
            # line3 : d(q_dot)/dall
            A3 = np.array([derivatives['velocity'][0]["angular_accel"][1], derivatives['velocity'][2]["angular_accel"][1],
                           derivatives['angular_vel'][1]["angular_accel"][1], derivatives['attitude'][0]["angular_accel"][1]
            ])
            # line4 : d(theta_dot)/dall
            A4 = np.array([derivatives['velocity'][0]["angle_der"][0], derivatives['velocity'][2]["angle_der"][0],
                           derivatives['angular_vel'][1]["angle_der"][0], derivatives['attitude'][0]["angle_der"][0]
            ])
            A_long = np.vstack((A1, A2, A3, A4))

            # Lateral
            # line1 d(v_dot)/dall
            A1 = np.array([derivatives['velocity'][1]["acceleration"][1], derivatives['angular_vel'][0]["acceleration"][1],
                           derivatives['angular_vel'][2]["acceleration"][1], derivatives['attitude'][1]["acceleration"][1]
            ])
            # line2 d(p_dot)/dall
            A2 = np.array([derivatives['velocity'][1]["angular_accel"][0], derivatives['angular_vel'][0]["angular_accel"][0],
                           derivatives['angular_vel'][2]["angular_accel"][0], derivatives['attitude'][1]["angular_accel"][0]
            ])
            # line3 d(r_dot)/dall
            A3 = np.array(
                [derivatives['velocity'][1]["angular_accel"][2], derivatives['angular_vel'][0]["angular_accel"][2],
                 derivatives['angular_vel'][2]["angular_accel"][2], derivatives['attitude'][1]["angular_accel"][2]
            ])
            # line4 d(phi_dot)/dall
            A4 = np.array(
                [derivatives['velocity'][1]["angle_der"][1], derivatives['angular_vel'][0]["angle_der"][1],
                 derivatives['angular_vel'][2]["angle_der"][1], derivatives['attitude'][1]["angle_der"][1]
            ])
            A_lat = np.vstack((A1, A2, A3, A4))

        else:
            raise NotImplementedError

        return A_long, A_lat

def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def wind2body4attitude(eps_v0, alpha, beta):
    """
    Fix for the fact that theta, phi, psi is the wrong axis order. To use rotation, it has to be phi, theta, psi
    """
    eps_vec = np.array([eps_v0[1], eps_v0[0], eps_v0[2]])
    eps_vec = wind2body(eps_vec, alpha, beta)
    return np.array([eps_vec[1], eps_vec[0], eps_vec[2]])

def body2wind4attitude(eps_v0, alpha, beta):
    """
    Fix for the fact that theta, phi, psi is the wrong axis order. To use rotation, it has to be phi, theta, psi
    """
    eps_vec = np.array([eps_v0[1], eps_v0[0], eps_v0[2]])
    eps_vec = body2wind(eps_vec, alpha, beta)
    return np.array([eps_vec[1], eps_vec[0], eps_vec[2]])

# TODO: numba jit
def _system_equations(time, state_vector, mass, inertia, forces, moments):
    """Euler flat earth equations: linear momentum equations, angular momentum
    equations, angular kinematic equations, linear kinematic
    equations.

    Parameters
    ----------
    time : float
        Current time (s).
    state_vector : array_like, shape(9)
        Current value of absolute velocity and angular velocity, both
        expressed in body axes, euler angles and position in Earth axis.
        (u, v, w, p, q, r, theta, phi, psi, x, y, z)
         (m/s, m/s, m/s, rad/s, rad/s rad/s, rad, rad, rad, m, m ,m).
    mass : float
        Current mass of the aircraft (kg).
    inertia : array_like, shape(3, 3)
        3x3 tensor of inertia of the aircraft (kg * m2)
        Current equations assume that the aircraft has a symmetry plane
        (x_b - z_b), thus J_xy and J_yz must be null.
    forces : array_like, shape(3)
        3 dimensional vector containing the total total_forces (including
        gravity) in x_b, y_b, z_b axes (N).
    moments : array_like, shape(3)
        3 dimensional vector containing the total total_moments in x_b,
        y_b, z_b axes (N·m).

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
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight", Courier Corporation,
        p. 149 (5.8 The Flat-Earth Approximation), 2012.

    .. [2] M. A. Gómez Tierno y M. Pérez Cortés, "Mecánica del Vuelo",
        Garceta Grupo Editorial, pp.18-25 (Tema 2: Ecuaciones Generales del
        Moviemiento), 2012.

    """
    # Note definition of total_moments of inertia p.21 Gomez Tierno, et al
    # Mecánica de vuelo
    Ix = inertia[0, 0]
    Iy = inertia[1, 1]
    Iz = inertia[2, 2]
    Jxz = - inertia[0, 2]

    Fx, Fy, Fz = forces
    L, M, N = moments

    u, v, w = state_vector[0:3]
    p, q, r = state_vector[3:6]
    theta, phi, psi = state_vector[6:9]

    # Linear momentum equations
    du_dt = Fx / mass + r * v - q * w
    dv_dt = Fy / mass - r * u + p * w
    dw_dt = Fz / mass + q * u - p * v

    # Angular momentum equations
    dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
             p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
    dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
    dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
             q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)

    # Angular Kinematic equations
    dtheta_dt = q * cos(phi) - r * sin(phi)
    dphi_dt = p + (q * sin(phi) + r * cos(phi)) * np.tan(theta)
    dpsi_dt = (q * sin(phi) + r * cos(phi)) / cos(theta)

    # Linear kinematic equations
    dx_dt = (cos(theta) * cos(psi) * u +
             (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) * v +
             (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * w)
    dy_dt = (cos(theta) * sin(psi) * u +
             (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) * v +
             (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) * w)
    dz_dt = -u * sin(theta) + v * sin(phi) * cos(theta) + w * cos(
        phi) * cos(theta)

    return np.array([du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dtheta_dt,
                     dphi_dt, dpsi_dt, dx_dt, dy_dt, dz_dt])
