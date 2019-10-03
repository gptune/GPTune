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
import numpy as np
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))


from gptune import *
from autotune.problem import *
from autotune.space import *
from autotune.search import *

################################################################################

# Define Problem

# Argmin{x} objective(t,x), for x in [0., 1.]

input_space = Space([Real(0., 10., name="t")])
parameter_space = Space([Real(0., 1., name="x")])
output_space = Space([Real(float('-Inf'), float('Inf'), name="time")])

def objective(point):

    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """

    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t 
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e

    return f

constraints = {"cst1" : "x >= 0. and x <= 1."}

problem = TuningProblem(input_space, parameter_space, output_space, objective, constraints, None)

# Run Autotuning

#search_param_dict = {}
#search_param_dict['method'] = 'MLA'

#search = Search(problem, search_param_dict)

#search.run()
def number_of_processes_and_threads(point):

    nproc = 1
    nth = 1

    return (nproc, nth)

computer = Computer(nodes = 1, cores = 1, hosts = None, number_of_processes_and_threads = number_of_processes_and_threads)
options = Options()
options['sample_algo'] = 'MCS'

gt = GPTune(problem, computer = computer, data = None, options = options)
(data, modeler) = gt.MLA(NS = 20, NI = 1, NS1 = 10)
print(data.Y)
print([(y[-1], min(y)[0], max(y)[0]) for y in data.Y])

