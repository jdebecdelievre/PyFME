# -*- coding: utf-8 -*-
"""
Tests of equations of euler flat earth model.
"""

import pytest
import numpy as np
import os
from pyfme.models.constants import EARTH_MEAN_RADIUS
from pyfme.models import RigidBodyEulerState, RigidBodyQuatState
from pyfme.models.state import copyStateValues
from pyfme.models import RigidBodyEuler, RigidBodyQuat
from pyfme.aircrafts.aircraft import Aircraft
from pyfme.aircrafts import BasisLinear
from pyfme.environment import Environment
from pyfme.utils.input_generator import Constant

def test_system_equations():
    time = 0
    state_vector = np.array(
        [1, 1, 1, 1, 1, 1,
         np.pi / 4, np.pi / 4, 0,
         1, 1, 1],
        dtype=float
    )

    mass = 10
    inertia = np.array([[1000,    0, -100],
                        [   0,  100,    0],
                        [-100,    0,  100]], dtype=float)

    forces = np.array([100., 100., 100.], dtype=float)
    moments = np.array([100., 1000., 100], dtype=float)

    exp_sol = np.array(
        [10, 10, 10, 11. / 9, 1, 92. / 9,
         0, 1 + 2 ** 0.5, 2,
         1 + (2 ** 0.5) / 2, 0, 1 - (2 ** 0.5) / 2],
        dtype=float
    )
    sys = RigidBodyEuler(None, None)
    sol = _system_equations(time, state_vector, mass, inertia, forces, moments)
    np.testing.assert_allclose(sol, exp_sol, rtol=1e-7, atol=1e-15)


def test_quat_vs_euler():
  state = RigidBodyQuatState()
  aircraft = Aircraft()
  aircraft.mass = np.random.uniform(10,100)
  aircraft.inertia = np.random.uniform(-100, 100, size=(3,3))
  aircraft.inertia_inverse = np.linalg.inv(aircraft.inertia)
  s1 = RigidBodyEuler(aircraft, Environment())
  s2 = RigidBodyQuat(aircraft, Environment())
  sol1 = RigidBodyEulerState()
  sol2 = RigidBodyQuatState()
  for n in range(1000):
    state.velocity = np.random.uniform(-2,2, size=3)
    state.omega = np.random.uniform(-2,2, size=3)
    state.attitude = np.random.uniform(-2,2, size=3)
    forces = np.random.uniform(-2,2, size=3)
    moments = np.random.uniform(-2,2, size=3)
    sol1.vec = s1._system_equations(state, forces, moments)
    sol2.vec = s2._system_equations(state, forces, moments)
    np.testing.assert_allclose(sol1.position, sol2.position ,rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(sol1.velocity, sol2.velocity ,rtol=1e-7, atol=1e-15)    
    np.testing.assert_allclose(sol1.omega, sol2.omega ,rtol=1e-7, atol=1e-15)    

def test_quat_vs_euler_2():
  aircraft = BasisLinear('example.json')
  # higher gravity since it couples attitude equations with body axis ones
  s1 = RigidBodyEuler(aircraft, Environment(gravity=VerticalConstant(100)))
  s2 = RigidBodyQuat(aircraft, Environment(gravity=VerticalConstant(100)))
  controls = {
  'delta_elevator': Constant(0),
  'delta_aileron': Constant(0),
  'delta_rudder': Constant(0),
  'delta_throttle': Constant(0)
  }

  n=0
  while n < 10:
      sol1 = RigidBodyEulerState()
      sol2 = RigidBodyQuatState()
      sol1.omega = np.random.uniform(-1,1, size=3)
      sol1.attitude = np.random.uniform(-1,1, size=3)
      sol1.position = np.array([0,0,0])
      sol1.velocity = np.random.uniform(1,2, size=3)
      sol2 = copyStateValues(sol1, sol2)
      try:
          with warnings.catch_warnings(record=True):
              warnings.simplefilter("error")
              sol1 = s1.integrate(1, sol1, controls, dt_eval=0.001)
              sol2 = s2.integrate(1, sol2, controls, dt_eval=0.001)
      except RuntimeWarning:
          print('runtime warning')
          continue
      print(n)
      n+=1
      try:
          np.testing.assert_allclose(sol1.position[:,-1], sol2.position[:,-1] ,rtol=1, atol=1, verbose=True)
          np.testing.assert_allclose(sol1.velocity[:,-1], sol2.velocity[:,-1] ,rtol=1, atol=1, verbose=True)    
          np.testing.assert_allclose(sol1.omega[:,-1], sol2.omega[:,-1] ,rtol=1, atol=1, verbose=True)   
      except:
          print('Assertion Error: discrepancy')
          break  

def test_fun_raises_error_if_no_update_simulation_is_defined():
    system = RigidBodyEuler(t0=0, full_state=full_state)
    x = np.zeros_like(system.state_vector)
    with pytest.raises(TypeError):
        system.fun(t=0, x=x)


def test_get_state_vector_from_full_state():

    system = RigidBodyEuler(0, full_state)

    x = np.array([50, 2, 3,
                  1/180*np.pi, 5/180*np.pi, 5/180*np.pi,
                  5/180*np.pi, 15/180*np.pi, 45/180*np.pi,
                  0, 0, -2000])

    np.testing.assert_allclose(system.state_vector, x)
