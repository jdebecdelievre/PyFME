"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Generic Aircraft
----------------

"""
from abc import abstractmethod

import numpy as np

from pyfme.utils.anemometry import tas2cas, tas2eas, calculate_alpha_beta_TAS


class Aircraft(object):

    def __init__(self):
        # Mass & Inertia
        self.mass = 0  # kg
        self.inertia = np.zeros((3, 3))  # kg·m²

        # Geometry
        self.Sw = 0  # m2
        self.chord = 0  # m
        self.span = 0  # m

        # Controls
        self.controls = {}
        self.control_limits = {}

        # Coefficients
        # Aero
        self.CL, self.CD, self.Cm = 0, 0, 0
        self.CY, self.Cl, self.Cn = 0, 0, 0

        # Thrust
        self.Ct = 0

        # Forces & moments
        self.total_forces = np.zeros(3)
        self.total_moments = np.zeros(3)

        # Velocities
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number
        self.q_inf = 0  # Dynamic pressure at infty (Pa)

        # Angles
        self.alpha = 0  # Angle of attack (AOA).
        self.beta = 0  # Angle of sideslip (AOS).
        self.alpha_dot = 0  # Rate of change of AOA.

    @property
    def Ixx(self):
        return self.inertia[0, 0]

    @property
    def Iyy(self):
        return self.inertia[1, 1]

    @property
    def Izz(self):
        return self.inertia[2, 2]

    @property
    def Fx(self):
        return self.total_forces[0]

    @property
    def Fy(self):
        return self.total_forces[1]

    @property
    def Fz(self):
        return self.total_forces[2]

    @property
    def Mx(self):
        return self.total_moments[0]

    @property
    def My(self):
        return self.total_moments[1]

    @property
    def Mz(self):
        return self.total_moments[2]

    def _set_current_controls(self, controls):

        # If a control is not given, the previous value is assigned.
        for control_name, control_value in controls.items():
            limits = self.control_limits[control_name]
            if limits[0] <= control_value <= limits[1]:
                self.controls[control_name] = control_value
            else:
                # TODO: maybe raise a warning and assign max deflection
                msg = (
                    f"Control {control_name} out of range ({control_value} "
                    f"when min={limits[0]} and max={limits[1]})"
                )
                raise ValueError(msg)

    def _calculate_aerodynamics(self, state, environment):

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = state.velocity.vel_body - environment.body_wind

        self.alpha, self.beta, self.TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2]
        )

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2

    def _calculate_aerodynamics_2(self, TAS, alpha, beta, environment):

        self.alpha, self.beta, self.TAS = alpha, beta, TAS

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2


    @abstractmethod
    def calculate_forces_and_moments(self, state, environment, controls):

        self._set_current_controls(controls)
        self._calculate_aerodynamics(state, environment)


    def calculate_derivatives(self, state, environment, controls, eps=1e-3):
        """
        Calculate dimensional derivatives of the forces at the vicinity of the state.
        The output consists in 2 dictionaries, one for force one for moment
        key: type of variables derivatives are taken for
        val : 3x3 np array with X,Y,Z and L,M,N as columns, and the variable we differentiate against in lines
        (u,v,w ; phi,theta,psi ; p,q,r ; x,y,z)
        """
        names = {'velocity': ['u', 'v', 'w'],
                 'angular_vel': ['p', 'q', 'r'],
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
