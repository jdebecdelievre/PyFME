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
from copy import deepcopy
from numba import jit

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

    @abstractmethod
    def make_state_obj(self, state_vec=None, **options):
        raise NotImplementedError

    # @jit
    def _fun(self, t, state_vec, controls_sequence):
        
        state = self.make_state_obj(state_vec)

        # get controls at time t
        controls = self.aircraft.get_controls(t, controls_sequence)

        # estimate aerodynamic conditions
        conditions = self.environment.calculate_aero_conditions(state)

        # estimate aero forces and moments
        forces, moments = self.aircraft.calculate_forces_and_moments(state,
                                    conditions, controls)

        # add gravity
        fg = conditions.gravity_vector * self.aircraft.mass

        # Get state variables
        return self._system_equations(state, fg + forces, moments)

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
        def fun(t, x):
            return self._fun(t, x, controls_sequence)

        # If a state object is input, the output will be the same state object
        if not isinstance(x0, np.ndarray):
            return_state = True
            x0 = 1.0*np.squeeze(x0.vec)

        # solve
        sol = solve_ivp(fun, t_span, x0, method=method, t_eval=t_eval,
                        dense_output=dense_output, vectorized=False,
                        **options)
        if sol.status != 0:
            print(sol.message)
        
        if return_state:
            state = self.make_state_obj(state_vec=sol.y, time=sol.t, aircraft=self.aircraft)
        else:
            state = pd.DataFrame(sol.y.T, columns=self.info)
            state['time'] = sol.t
        return state

    @abstractmethod
    def trim(self):
        raise NotImplementedError


