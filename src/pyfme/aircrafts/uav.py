# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
----------
Hypothetical Fixed Wing UAV - AVL is ran to get forces and moments
----------
"""

import numpy as np
nl = np.linalg
import pdb
from scipy.interpolate import RectBivariateSpline
from scipy.stats import linregress

from pyfme.aircrafts.aircraft import Aircraft
from pyfme.models.constants import slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body, body2wind
from copy import deepcopy as cp
from pyfme.aero.avl import avl_run
import pandas as pd
import matplotlib.pyplot as plt


class MeterSpanUAV(Aircraft):
    """
    """

    def __init__(self, avl_file):

        # AVL stuff
        # self.avl = avl_run(avl_geometry_file, )
        self.data = pd.read_pickle(avl_file)
        self._build_aero_model()

        # Mass & inertia
        self.mass = .50
        self.inertia = np.diag((.01, 0.02, 0.01))
        self.cg = [.3, 0, 0]

        # Reference values
        self.Sw = .9**2/6  # m2
        self.span = .9  # m
        self.chord = self.Sw/ self.span  # m
        self.propeller_radius = 0  # m

        # CONTROLS
        self.controls = {'delta_elevator': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}

        self.control_limits = {'delta_elevator': (np.deg2rad(-26),
                                                  np.deg2rad(28)),  # rad
                               'delta_aileron': (np.deg2rad(-15),
                                                 np.deg2rad(20)),  # rad
                               'delta_rudder': (np.deg2rad(-16),
                                                np.deg2rad(16)),  # rad
                               'delta_t': (0, 1)}  # non-dimensional

        # Aerodynamic Coefficients
        self.CL, self.CD, self.Cm = 0, 0, 0
        self.CY, self.Cl, self.Cn = 0, 0, 0

        # Thrust Coefficient
        self.Ct = 0

        self.total_forces = np.zeros(3)
        self.total_moments = np.zeros(3)

        # Velocities
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number
        self.q_inf = 0  # Dynamic pressure at infinity (Pa)

        # Angles
        self.alpha = 0  # rad
        self.beta = 0  # rad
        self.alpha_dot = 0  # rad/s

    def _build_aero_model(self):
        self.CL_0 = 0.148
        self.CM_0 = 0.0075
        self.CL_alpha = 5.440E+00
        self.CL_q = np.mean(self.data.CLq)
        self.CL_delta_elev = np.sum(self.data.de*self.data.CLd1)/np.sum(self.data.de**2)

        self.CM_alpha2, self.CM_alpha, self.CM_0 = np.polyfit(self.data.alpha, self.data.Cm, 2)
        self.CM_q = 2*np.mean(self.data.Cmq)
        self.CM_delta_elev = np.sum(self.data.de*self.data.Cmd1)/np.sum(self.data.de**2)

        # pre-stall drag model
        ICL_max = self.data.CL.idxmax()
        cl = self.data.CL[:ICL_max-2]
        cd = self.data.CD[:ICL_max-2]
        al = self.data.alpha[: ICL_max]
        self.CD_K1, self.CD_0, r_value, p_value, std_err = linregress(cl ** 2, cd)
        self.CL_MAX = self.data.CL[ICL_max]

        self.CY_beta = np.mean(self.data.CYb)
        self.CY_p = np.mean(self.data.CYp)
        self.CY_r = np.mean(self.data.CYr)
        self.CY_delta_rud = np.mean(self.data.dr)

        self.Cl_beta = np.mean(self.data.Clb)
        self.Cl_p = np.mean(self.data.Clp)
        self.Cl_r_cl = np.sum(self.data.CL*self.data.Clr)/np.sum(self.data.CL**2)
        self.Cl_delta_rud = np.mean(self.data.Cld2)
        self.Cl_delta_aile = np.sum(self.data.da*self.data.Cld2)/np.sum(self.data.da**2)

        self.CN_beta = np.mean(self.data.Cnb)
        self.CN_p_al = np.sum(self.data.alpha*self.data.Cnp)/np.sum(self.data.alpha**2)
        self.CN_r_cl, self.CN_r_0, _, _,_ = linregress(self.data.CL**2,self.data.Cnr)
        self.CN_delta_rud = np.mean(self.data.Cnd3)
        # x = np.reshape(self.data.CL, (1, 12)) * np.reshape(self.data.da, (9, 1))
        # self.CN_delta_aile_cl = np.sum(self.data.Cnd2*x) / np.sum(x**2)

    @property
    def delta_elevator(self):
        return self.controls['delta_elevator']

    @property
    def delta_rudder(self):
        return self.controls['delta_rudder']

    @property
    def delta_aileron(self):
        return self.controls['delta_aileron']

    @property
    def delta_t(self):
        return self.controls['delta_t']

    def calculate_aero_coeffs(self, state, controls):
        # Compute features
        V = nl.norm(state.velocity.vel_body)
        p = state.angular_vel.p * self.span / (2*V)
        q = state.angular_vel.q * self.chord / (2*V)
        r = state.angular_vel.r * self.span / (2*V)
        alpha = np.arctan2(state.velocity.w, state.velocity.u)
        beta = np.arcsin(state.velocity.v, V)

        # Run AVL
        avl_state = np.array([alpha, beta, p, q, r])
        avl_controls = np.array([self.delta_elevator, self.delta_aileron, self.delta_rudder])
        data = self.avl.run(avl_state, avl_controls)

        # set values for non-dimensional coefficients.
        for attr in data.columns:
            setattr(self, attr, data[attr][0])

    def _calculate_aero_forces_moments(self, state):
        q = self.q_inf
        Sw = self.Sw
        c = self.chord
        b = self.span

        self.calculate_aero_coeffs(state)

        L = q * Sw * self.CL
        D = q * Sw * self.CD
        Y = q * Sw * self.CY
        l = q * Sw * b * self.Cl
        m = q * Sw * c * self.Cm
        n = q * Sw * b * self.Cn

        return L, D, Y, l, m, n

    def _calculate_thrust_forces_moments(self, environment):
        return np.array([1, 0, 0]), np.array([0, 0, 0])

    def calculate_forces_and_moments(self, state, environment, controls):
        # Update controls and aerodynamics
        super().calculate_forces_and_moments(state, environment, controls)

        Ft, Mt = self._calculate_thrust_forces_moments(environment)
        L, D, Y, l, m, n = self._calculate_aero_forces_moments(state)
        Fg = environment.gravity_vector * self.mass

        Fa_wind = np.array([-D, Y, -L])
        Fa_body = wind2body(Fa_wind, self.alpha, self.beta)
        Fa = Fa_body

        self.total_forces = Ft + Fg + Fa
        self.total_moments = np.array([l, m, n])

        self.Fa_wind = Fa_wind

        # return state.velocity._vel_body, state.angular_vel._vel_ang_body
        return self.total_forces, self.total_moments



if __name__=='__main__':
    a = MeterSpanUAV('MeterSpanUAV.pkl')
    plt.plot(a.data.CL, a.data.alpha)
    plt.show()

