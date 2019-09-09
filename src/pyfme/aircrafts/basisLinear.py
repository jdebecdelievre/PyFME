import numpy as np
import pdb
import json

from pyfme.aircrafts.aircraft import Aircraft, ConventionalControls
from pyfme.models.constants import slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body, body2wind
from copy import deepcopy as cp
from collections import namedtuple
from pyfme.environment import Conditions
import os
pth = os.path.dirname(os.path.realpath(__file__))

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

        self.AR = self.span**2/self.Sw



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
            self.CL_alpha * alpha_RAD +
            # self.CL_alpha*np.sin(2*np.pi*alpha_RAD)/2/np.pi +
            self.CL_delta_elev*delta_elev +
            self.CL_q * q * c/(2*V)
        )
        # STALL
        # CL = CL * (abs(CL) < self.CL_MAX) + np.sign(CL)*self.CL_MAX*(1-(abs(CL) < self.CL_MAX))

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
        # CL = CL * (abs(CL) < self.CL_MAX) + np.sign(CL)*self.CL_MAX*(1-(abs(CL) < self.CL_MAX))

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


class Linear(Aircraft):

    def __init__(self, aircraft_file=None, alpha_dot=False):

        super().__init__()
        
        aircraft_file = os.path.join(pth, 'linear', 'linear.json') if aircraft_file is None else aircraft_file

        # set to loaded values
        with open(aircraft_file, 'r') as f:
        	aircraft_file = json.load(f)
        for d in aircraft_file:
        	setattr(self, d, aircraft_file[d])

        if alpha_dot == False:
            self.CL_alpha_dot = 0.0
            self.CM_alpha_dot = 0.0
        self.state_dot = None
        self.AR = self.span**2/self.Sw
        self.store = []

        self.inertia_inverse = np.linalg.inv(self.inertia)

    def get_controls(self, t, controls_sequence):
        return ConventionalControls().evaluate_sequence(t, controls_sequence)

    def _calculate_aero_lon_forces_moments_coeffs(self, alpha, V, state, controls):
        delta_elev = controls.delta_elevator # deg
        c = self.chord  # m
        p, q, r = state.omega.T # rad/s
        D_alpha = alpha - self.alpha_0

        state_dot = self.state_dot # trick to avoid having to extend state space
        if self.state_dot == None:
            alpha_dot = 0.
        else:
            alpha_dot = (self.state_dot.w * state.u - self.state_dot.u * state.w) / (state.u**2 + state.w**2)
            # self.state_dot = None
            self.store.append(alpha_dot)
            # alpha_dot = 0
        CL = ( 
             self.CL_0 +
             self.CL_alpha * D_alpha +
             self.CL_delta_elev * delta_elev +
             self.CL_q * q * c/(2*V) + 
             self.CL_alpha_dot * alpha_dot
        )

        CD = (
            self.CD_0 + 2/(self.AR*self.e*np.pi)*CL*self.CL_alpha*D_alpha
        )

        CM = (
             self.CM_0 + 
             self.CM_alpha * D_alpha +
             self.CM_delta_elev * delta_elev +
             self.CM_q * q * c/(2*V) + 
             self.CM_alpha_dot * alpha_dot
        )
        return CL, CD, CM

    def _calculate_aero_lat_forces_moments_coeffs(self, alpha, beta, V, state, controls):
        delta_aile = controls.delta_aileron # deg
        delta_rud = controls.delta_rudder  # deg
        delta_elev = controls.delta_elevator # deg
        b = self.span
        p, q, r = state.omega.T
        D_alpha = alpha - self.alpha_0

        # Recompute CL\        
        state_dot = self.state_dot # trick to avoid having to extend state space
        if self.state_dot == None:
            alpha_dot = 0.
        else:
            alpha_dot = (self.state_dot.w * state.u - self.state_dot.u * state.w) / (state.u**2 + state.w**2)
        # CL = self.CL
        CL = ( 
             self.CL_0 +
             self.CL_alpha * D_alpha +
             self.CL_delta_elev * delta_elev +
             self.CL_q * q * self.chord/(2*V) + 
             self.CL_alpha_dot * alpha_dot
        )

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

        CN = (
            self.CN_beta * beta +
            self.CN_delta_aile * delta_aile +
            self.CN_delta_rud * delta_rud +
            b/(2 * V) * (self.CN_p * p + self.CN_r * r)
        )
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