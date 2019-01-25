# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Export results and state to other formats
----------------
Creates a few functions to save simulation outputs to other formats
Other functions and methods doing the same thing:
    AircraftState.save_to_json()
"""
from scipy.io import savemat


def results2matlab(results, filename):
    ref = results.to_dict(orient='list')
    savemat(filename, ref)
