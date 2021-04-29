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

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import *
import numpy as np
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')


    args = parser.parse_args()

    return args

def demo_func(t, x):

    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1

    return f

def objectives(point):
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
    f = d * e + 1

    # print('test:',test)
    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f]

def main():

    import matplotlib.pyplot as plt

    args = parse_args()
    ntask = args.ntask

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['model_class'] = 'Model_LCM'

   # giventask = [[6],[6.5]]
    #giventask = [[i] for i in np.arange(0, ntask/2, 0.5).tolist()]
    giventask = [[1.0],[0.5]]
    print ("giventask: ", giventask)

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
    (models, model_function) = gt.LoadSurrogateModel(Igiven = giventask, method = "max_evals")

    " A quick validation"
    print (model_function({"t": 1.0, "x": 0.05}))
    (mu, var) = models[0].predict(np.array([0.05]), 0)
    print ("GP model")
    print (mu, var)

#    for tid in [0, 1]:
#        for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
#            for o in range(1):
#                print ("tid: ", tid)
#                print ("p: ", p)
#                (mu, var) = models[o].predict(np.array([p]), tid)
#                print ("GP model")
#                print (mu, var)
#                print ("True function value")
#                print (demo_func(tid, p))

if __name__ == "__main__":
    main()
