#! /usr/bin/env python

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

input_space = Space([Real(0., 10., name="x")])
parameter_space = Space([Real(0., 1., name="t")])
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

constraints = {"cst1" : "x >= 0. and x <= 1."}

problem = TuningProblem(input_space, parameter_space, output_space, objective, constraints, None)

# Run Autotuning

#search_param_dict = {}
#search_param_dict['method'] = 'MLA'

#search = Search(problem, search_param_dict)

#search.run()
gt = GPTune(problem)
gt.MLA()

