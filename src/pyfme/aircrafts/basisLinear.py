# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
----------
Basis Linear Model
----------

References
----------
[1] ROSKAM, J., Methods for Estimating Stability and Control
    Derivatives of Conventional Subsonic Airplanes
[2] McDonnell Douglas Co., The USAF and Stability and Control
    Digital DATCOM, Users Manual
----------

CD            - drag coefficient as fhunction of the angle of attack via [2]
CD_delta_elev - incremental induced-drag coefficient as function of the angle
                of attack and deflection angle via [2]

CLsus            - lift coefficient as a function of the angle of attack
                   via [2]
CLsus_alphadot   - lift coefficient derivative with respect to the angle
                   of attack acceleration as function of the angle of attack
                   via [2]
CLsus_q          - lift coefficient derivative with respect to the pitch rate
                   via [2]
CLsus_delta_elev - lift coefficient derivative with respect to the elevator
                   deflection as function of the angle of attack via [2]

CY_beta      - side force coefficient derivative with respect to the angle of
               sideslip via [2]
CY_p         - side force coefficient derivative with respect to the roll rate
               as function of the angle of attack via [2]
CY_r         - side force coefficient derivative with respect to the yaw rate
               as function of the angle of attack via [1]
CY_delta_rud - side force coefficient derivative with respect to the rudder
               deflection via [1]

Cl_beta       - rolling moment coefficient derivative with respect to the
                sideslip angle as function of the angle of attack via [2]
Cl_p          - rolling moment coefficient derivative with respect to the roll
                rate as function of the angle of attack via [2]
Cl_r          - rolling moment coefficient derivative with respect to the yaw
                rate as function of the angle of attack via [2]
Cl_delta_rud  - rolling moment coefficient derivative with respect to the
                rudder deflection as function of the angle of attack via [1]
Cl_delta_aile - incremental rolling moment coefficient derivative with
                respect to the aileron deflection as function of the aileron
                deflection via [2]

CM            - pitching moment coefficient derivative with respect to the
                angle of attack as function of the angle of attack via [2]
CM_q          - pitching moment coefficient derivative with respect to the
                pitch rate via [2]
CM_alphadot   - pitching moment coefficient derivative with respect to the
                acceleration of the angle of attack as function of the angle
                of attack via [2]
CM_delta_elev - pitching moment coefficient derivative with respect to the
                elevator deflection as function of the elevator deflection
                via [2]

CN_beta       - yawing moment coefficient derivative with respect to the
                sideslip angle via [2]
CN_p          - yawing moment coefficient derivative with respect to the roll
                rate as a function of the angle of attack via [2]
CN_r          - yawing moment coefficient derivative with respect to the yaw
                rate as a function of the angle of attack via [2]
CN_delta_rud  - yawing moment coefficient derivative with respect to the
                rudder deflection as function of the angle of attack via [1]
CN_delta_aile - incremental yawing moment coefficient derivative with respect
                to the aileron deflection as a function of the aileron
                deflection and the angle of attack via [2]
"""
import numpy as np
import pdb
import json

from pyfme.aircrafts.aircraft import Aircraft, ConventionalControls
from pyfme.models.constants import slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body, body2wind
from copy import deepcopy as cp
from collections import namedtuple
from pyfme.environment import Conditions

inertia_attributes=[
'mass',
'inertia']

aero_attributes = [
'CL_0',
'CM_0',
'CL_alpha',
'CL_q',
'CL_delta_elev',
'CM_alpha2',
'CM_alpha',
'CM_q',
'CM_delta_elev',
'e',
'CD_0',
'CL_MAX',
'CY_beta',
'CY_p',
'CY_r',
'CY_delta_rud',
'Cl_beta',
'Cl_p',
'Cl_r',
'Cl_delta_rud',
'Cl_delta_aile',
'CN_beta',
'CN_p_al',
'CN_r_cl',
'CN_r_0',
'CN_delta_rud',
'CN_delta_aile']

geometrical_attributes =[
'Sw',
'chord',
'span'
]

class BasisLinear(Aircraft):
    """
    Cessna 172
    The Cessna 172 is a blablabla...
    """

    def __init__(self, aircraft_file):

        super().__init__()

        # set to 0 by default
        for d in aero_attributes+geometrical_attributes+inertia_attributes:
        	setattr(self, d, 0)

        # set to loaded values
        with open(aircraft_file, 'r') as f:
        	aircraft_file = json.load(f)
        for d in aircraft_file:
        	setattr(self, d, aircraft_file[d])
        self.inertia_inverse = np.linalg.inv(self.inertia)



    def get_controls(self, t, controls_sequence):
        return ConventionalControls().evaluate_sequence(t, controls_sequence)

    def _calculate_aero_lon_forces_moments_coeffs(self, alpha, V, state, controls):
        """
        Simplified dynamics for the Cessna 172: strictly linear dynamics.
        Stability derivatives are considered constant, the value for small angles is kept.

        Parameters
        ----------
        state

        Returns
        -------

        """

        delta_elev = controls.delta_elevator
        alpha_RAD = alpha # rad
        c = self.chord  # m
        p, q, r = state.omega.T  # rad/s
        CL = (
            self.CL_0 +
            self.CL_alpha*alpha_RAD +
            self.CL_delta_elev*delta_elev +
            self.CL_q * q * c/(2*V)
        )
        # STALL
        CL = CL * (abs(CL) < self.CL_MAX) + np.sign(CL)*self.CL_MAX*(1-(abs(CL) < self.CL_MAX))

        CD = self.CD_0 + CL**2/(self.AR*self.e*np.pi)

        CM = (
            self.CM_0 +
            (self.CM_alpha2*alpha + self.CM_alpha)*alpha +
            self.CM_delta_elev * delta_elev +
            self.CM_q * q * c/(2*V)
        )
        return CL, CD, CM

    def _calculate_aero_lat_forces_moments_coeffs(self, alpha, beta, V, state, controls):
        delta_aile = controls.delta_aileron # rad
        delta_rud = controls.delta_rudder  # rad
        b = self.span
        p, q, r = state.omega.T

        # Recompute CL
        delta_elev = np.rad2deg(controls.delta_elevator)
        CL = (
            self.CL_0 +
            self.CL_alpha*alpha +
            self.CL_delta_elev*delta_elev +
            self.CL_q * q * self.chord/(2*V)
        )
        CL = CL * (abs(CL) < self.CL_MAX) + np.sign(CL)*self.CL_MAX*(1-(abs(CL) < self.CL_MAX))

        CY = (
            self.CY_beta * beta +
            self.CY_delta_rud * delta_rud +
            b/(2 * V) * (self.CY_p * p + self.CY_r * r)
        )
        Cl = (
            self.Cl_beta * beta +
            self.Cl_delta_aile * delta_aile +
            self.Cl_delta_rud * delta_rud +
            b/(2 * V) * (self.Cl_p * p + self.Cl_r * r)
        )
            # b/(2 * V) * (self.Cl_p * p + self.Cl_r_cl * CL * r)

        CN = (
            self.CN_beta * beta +
            (self.CN_delta_aile*delta_aile) +
            self.CN_delta_rud * delta_rud +
            b/(2 * V) * (self.CN_p_al*alpha * p + (self.CN_r_cl*CL**2 + self.CN_r_0) * r)
        )
            # b/(2 * V) * (self.CN_p_al*alpha_DEG * p + (self.CN_r_cl*CL**2 + self.CN_r_0) * r)
        return CY, Cl, CN

    def _calculate_thrust_forces_moments(self, TAS, conditions, controls):
    	return 0

    def _calculate_aero_forces_moments(self, conditions, state, controls):
        q_inf = conditions.q_inf
        V = conditions.TAS
        alpha = conditions.alpha
        beta = conditions.beta

        Sw = self.Sw
        c = self.chord
        b = self.span

        CL, CD, Cm = self._calculate_aero_lon_forces_moments_coeffs(alpha, V, state, controls)
        CY, Cl, Cn = self._calculate_aero_lat_forces_moments_coeffs(alpha, beta, V, state, controls)

        L = q_inf * Sw * CL
        D = q_inf * Sw * CD
        Y = q_inf * Sw * CY
        l = q_inf * Sw * b * Cl
        m = q_inf * Sw * c * Cm
        n = q_inf * Sw * b * Cn

        return L, D, Y, l, m, n

    def calculate_derivatives(self, state, environment, controls, eps=1e-3):
        """
        Calculate dimensional derivatives of the forces at the vicinity of the state.
        The output consists in 2 dictionaries, one for force one for moment
        key: type of variables derivatives are taken for
        val : 3x3 np array with X,Y,Z and L,M,N as columns, and the variable we differentiate against in lines
        (u,v,w ; phi,theta,psi ; p,q,r ; x,y,z)
        """
        names = {'velocity': ['u', 'v', 'w'],
                 'omega': ['p', 'q', 'r'],
                 'acceleration': ['w_dot']}
        Fnames = ['X', 'Y', 'Z']
        Mnames = ['L', 'M', 'N']

        # F, M = self.calculate_forces_and_moments(state, environment, controls)

        # Rotation for stability derivatives in stability axis
        V = np.sqrt(state.velocity.u**2 + state.velocity.v**2 + state.velocity.w**2)
        alpha = np.arctan2(state.velocity.w, state.velocity.u)
        beta = np.arcsin(state.velocity.v / V)


        derivatives = {}
        for keyword in names.keys():
            for i in range(len(names[keyword])):
                eps_v0 = np.zeros(3)

                # plus perturb
                eps_v0[i] = eps/2
                eps_vec = wind2body(eps_v0, alpha, beta)
                state.perturbate(eps_vec, keyword)
                forces_p, moments_p = self.calculate_forces_and_moments(state, environment, controls)
                forces_p = body2wind(forces_p, alpha, beta)
                moments_p = body2wind(moments_p, alpha, beta)
                state.cancel_perturbation()

                # minus perturb
                eps_v0[i] = - eps/2
                eps_vec = wind2body(eps_v0, alpha, beta)
                state.perturbate(eps_vec, keyword)
                forces_m, moments_m = self.calculate_forces_and_moments(state, environment, controls)
                forces_m = body2wind(forces_m, alpha, beta)
                moments_m = body2wind(moments_m, alpha, beta)
                state.cancel_perturbation()

                k = names[keyword][i]
                for j in range(3):
                    # print(Fnames[j] + k, forces[j])
                    derivatives[Fnames[j] + k] = (forces_p[j] - forces_m[j]) / eps
                    derivatives[Mnames[j] + k] = (moments_p[j] - moments_m[j]) / eps

        return derivatives


    