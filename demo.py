#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department 
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


################################################################################

import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from gptune import *

import collections
from autotune.problem import *
from autotune.space import *
from autotune.search import *

################################################################################

# Define Problem

input_space = Space([Real(0., 10., name="t")])
parameter_space = Space([Real(0., 1., name="x"), Real(0., 1., name="y")])
output_space = Space([Real(float('-Inf'), float('Inf'), name="time")])

def objective(point):

    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t 
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e

    #return -0.5 * np.exp(-t1*(i1 - 2)**2) - 0.5 * np.exp(-t1 * (i1 + 2.1)**2 / 5) + 0.3
    return f

constraints = {"cst1" : "x >= 0. and y <= 1."}

problem = TuningProblem(input_space, parameter_space, output_space, objective, constraints, None)

# Run Autotuning

#search_param_dict = {}
#search_param_dict['method'] = 'MLA'

#search = Search(problem, search_param_dict)

#search.run()
gt = GPTune(problem)
(data, modeler) = gt.MLA(NS = 20, NI = 1, NS1 = 10)
print(data.Y)
print([(y[-1], min(y)[0], max(y)[0]) for y in data.Y])

