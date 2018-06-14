# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic System & Aircraft Dynamic System
----------------------------------------

Dynamic system class to integrate initial value problems numerically serves
as base for implementation of dynamic systems.

The Aircraft Dynamic Systems extends the Dynamic System taking into account
the Aircraft State.
"""
from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
from json import dump, load
import pandas as pd

class AircraftDynamicSystem:
    def __init__(self, aircraft, environment):
        """Aircraft Dynamic system initialization.

        Parameters
        ----------
        environment : Environment
            Environment Object
        aircraft : Aircraft
            Aircraft object
        """
        self.aircraft = aircraft
        self.environment = environment

    @abstractmethod
    def _system_equations(self, t, state_vec, controls_sequence):
        raise NotImplementedError

    def integrate(self, t_end, x0, controls_sequence, t_ini=0, dense_output=False,
                  dt_eval=0.01, method='RK45', options=None):
        """Integrate the system from current time to t_end.

        Parameters
        ----------
        t_end : float
            Final time.
        x0 : array_like or object
            Initial state vector or Object.
        dense_output: bool, opt
            Whether to compute a continuous solution. Default is True.
        dt_eval : float
            spacing between points
        method : str, opt
            Integration method. Accepts any method implemented in
            scipy.integrate.solve_ivp.
        options : dict
            Options for the selected method.

        Returns
        -------
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Solution values at t.
        sol : Bunch object with the following fields defined:
            t : ndarray, shape (n_points,)
                Time points.
            y : ndarray, shape (n, n_points)
                Solution values at t.
            sol : OdeSolution or None
                Found solution as OdeSolution instance, None if dense_output
                was set to False.
            t_events : list of ndarray or None
                Contains arrays with times at each a corresponding event was
                detected, the length of the list equals to the number of
                events. None if events was None.
            nfev : int
                Number of the system rhs evaluations.
            njev : int
                Number of the Jacobian evaluations.
            nlu : int
                Number of LU decompositions.
            status : int
                Reason for algorithm termination:
                -1: Integration step failed.
                0: The solver successfully reached the interval end.
                1: A termination event occurred.
            message : string
                Verbal description of the termination reason.
            success : bool
            True if the solver reached the interval end or a termination event
             occurred (status >= 0).
        """
        if options == None:
            options = {'max_step': dt_eval} # max step is distance between 2 samples

        # define time options
        t_span = (t_ini, t_end)
        t_eval = np.arange(t_ini, t_end, dt_eval)

        # define function to integrate
        def _fun(t, x):
            return self._system_equations(t, x, controls_sequence)

        # get vector state if needed
        if not isinstance(x0, np.ndarray):
            x0 = np.copy(x0.vec)

        # solve
        sol = solve_ivp(_fun, t_span, x0, method=method, t_eval=t_eval,
                        dense_output=dense_output, vectorized=False,
                        **options)
        if sol.status != 0:
            print(sol.message)

        return pd.DataFrame(sol.y.T, columns=self.info, index=sol.t)

    @abstractmethod
    def trim(self):
        raise NotImplementedError


class BodyAxisState:
    def __init__(self, state_vec=-np.ones(12), from_json=None):
        self.info = ["x_earth","y_earth","z_earth", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]
        # transpose if the state is [n,m], we want it [m,n] so that state_vec[1] is one state example
        if state_vec.ndim > 1:
            if state_vec.shape[1] != 12:
                state_vec = state_vec.T
        else:
            state_vec = np.expand_dims(state_vec, axis=0)
        self.vec = state_vec
        if from_json != None:
            self.load_from_json(from_json)

    @property
    def V(self):
        return np.linalg.norm(self.body_vel, axis=1)

    @property
    def N(self):
        # number of time steps are stored in this state object (for vectorization)
        return self.vec.shape[0]

    # Bunch of names for earth coordinates
    @property
    def earth_coordinates(self):
        return self.vec.T[:3]
    @earth_coordinates.setter
    def earth_coordinates(self, value):
        self.vec.T[:3] = value

    @property
    def x_earth(self):
        return self.vec.T[0]
    @x_earth.setter
    def x_earth(self, value):
        self.vec.T[0] = value

    @property
    def y_earth(self):
        return self.vec.T[1]
    @y_earth.setter
    def y_earth(self, value):
        self.vec.T[1] = value

    @property
    def z_earth(self):
        return self.vec.T[2]
    @z_earth.setter
    def z_earth(self, value):
        self.vec.T[2] = value

    @property
    def height(self):
        return -self.vec.T[2]

    # Bunch of names for earth coordinates
    @property
    def euler_angles(self):
        return self.vec.T[3:6]
    @euler_angles.setter
    def euler_angles(self, value):
        self.vec.T[3:6] = value

    @property
    def phi(self):
        return self.vec.T[3]
    @phi.setter
    def phi(self, value):
        self.vec.T[3] = value

    @property
    def theta(self):
        return self.vec.T[4]
    @theta.setter
    def theta(self, value):
        self.vec.T[4] = value

    @property
    def psi(self):
        return self.vec.T[5]
    @psi.setter
    def psi(self, value):
        self.vec.T[5] = value

    # Bunch of names for body_velocity
    @property
    def body_vel(self):
        return self.vec.T[6:9]
    @body_vel.setter
    def body_vel(self, value):
        self.vec.T[6:9] = value

    @property
    def u(self):
        return self.vec.T[6]
    @u.setter
    def u(self, value):
        self.vec.T[6] = value

    @property
    def v(self):
        return self.vec.T[7]
    @v.setter
    def v(self, value):
        self.vec.T[7] = value

    @property
    def w(self):
        return self.vec.T[8]
    @w.setter
    def w(self, value):
        self.vec.T[8] = value

    # Bunch of names for angular_velociy
    @property
    def euler_ang_rate(self):
        return self.vec.T[9:12]
    @euler_ang_rate.setter
    def euler_ang_rate(self, value):
        self.vec.T[9:12] = value

    @property
    def p(self):
        return self.vec.T[9]
    @p.setter
    def p(self, value):
        self.vec.T[9] = value

    @property
    def q(self):
        return self.vec.T[10]
    @q.setter
    def q(self, value):
        self.vec.T[10] = value

    @property
    def r(self):
        return self.vec.T[11]
    @r.setter
    def r(self, value):
        self.vec.T[11] = value

    def __repr__(self):
        rv = (
            "Aircraft State \n"
            f"{self.earth_coordinates} \n"
            f"{self.euler_angles} \n"
            f"{self.body_vel} \n"
            f"{self.euler_ang_rate} \n"
        )
        return rv

    def save_to_json(self, filename):
        state = dict()

        state['x_e'] = self.x_earth
        state['y_e'] = self.y_earth
        state['z_e'] = self.z_earth

        state['phi'] = self.phi
        state['theta'] = self.theta
        state['psi'] = self.psi

        state['u'] = self.u
        state['v'] = self.v
        state['w'] = self.w

        state['p'] = self.p
        state['q'] = self.q
        state['r'] = self.r

        with open(filename, 'w') as f:
            dump(state, f)

    def load_from_json(self, filename):
        self.vec = np.zeros(12)
        with open(filename, 'r') as f:
            state = load(f)
        for c_name, c_fun in state.items():
            try:
                setattr(self, c_name, c_fun)
            except AttributeError:
                print(f"Wrong argument name {c_name}")

