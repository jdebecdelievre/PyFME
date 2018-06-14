# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs generator
----------------
Provides some typical inputs signals such as: step, doublet, ramp, harmonic.
"""
from abc import abstractmethod

from numpy import vectorize, float64
from numpy import sin, pi
import numpy as np


def vectorize_float(method):
    vect_method = vectorize(method, otypes=[float64])

    def wrapper(self, *args, **kwargs):
        return vect_method(self, *args, **kwargs)

    return wrapper


# TODO: documentation
class Control(object):

    @abstractmethod
    def _fun(self, t):
        raise NotImplementedError

    def __call__(self, t):
        r = self._fun(t)
        return np.squeeze(r)

    def __add__(self, other):
        control = Control()
        control._fun = lambda t: self(t) + other(t)
        control._vec_fun = vectorize(control._fun, otypes=[float64])
        return control

    def __sub__(self, other):
        control = Control()
        control._fun = lambda t: self(t) - other(t)
        control._vec_fun = vectorize(control._fun, otypes=[float64])
        return control

    def __mul__(self, other):
        control = Control()
        control._fun = lambda t: self(t) * other(t)
        control._vec_fun = vectorize(control._fun, otypes=[float64])
        return control


class Constant(Control):

    def __init__(self, offset=0):
        self.offset = offset

    def _fun(self, t):
        return np.ones_like(t)*self.offset


class Step(Control):

    def __init__(self, t_init, T, A, offset=0):
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.t_fin = self.t_init + self.T

    def _fun(self, t):
        value = self.offset
        value = value + self.A*((self.t_init <= t) * (t <= self.t_fin))
        return value


class Doublet(Control):

    def __init__(self, t_init, T, A, offset=0):
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.t_fin1 = self.t_init + self.T / 2
        self.t_fin2 = self.t_init + self.T

    def _fun(self, t):
        value = self.offset + self.A / 2*((self.t_init <= t) * (t < self.t_fin1))\
                - self.A / 2*((self.t_fin1 < t) * (t <= self.t_fin2))
        return value


class Ramp(Control):

    def __init__(self, t_init, T, A, offset=0):
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.slope = self.A / self.T
        self.t_fin = self.t_init + self.T

    def _fun(self, t):
        value = self.offset + self.slope * (t - self.t_init) * \
                              ((self.t_init <= t) * (t <= self.t_fin))
        return value


class Harmonic(Control):

    def __init__(self, t_init, T, A, freq, phase=0, offset=0):
        super().__init__()
        self.t_init = t_init
        self.t_fin = t_init + T
        self.A = A
        self.freq = freq
        self.phase = phase
        self.offset = offset

    def _fun(self, t):
        value = self.offset + (self.A/2 * sin(2 * pi * self.freq * (t - self.t_init) +
                                    self.phase)) * ((self.t_init <= t) * (t <= self.t_fin))
        return value
