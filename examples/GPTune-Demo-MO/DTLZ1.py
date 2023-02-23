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

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import *

import argparse
import numpy as np
import time

import math

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')

    args = parser.parse_args()

    return args

def objectives(point):

    # DTLZ1 problem
    # We use the mathmatical model obtained from Pymoo test programs
    # (https://pymoo.org/problems/many/dtlz.html)

    x1 = point["x1"]
    x2 = point["x2"]
    x3 = point["x3"]

    #g_xm = 100 *(3 +\
    #        ((x1-0.5)**2 - np.cos(20*3.141592*(x1-0.5))) +\
    #        ((x2-0.5)**2 - np.cos(20*3.141592*(x2-0.5))) +\
    #        ((x3-0.5)**2 - np.cos(20*3.141592*(x3-0.5))))
    g_xm = 100 * (1 + ((x3-0.5)**2 - np.cos(20*3.141592*(x3-0.5))))

    f1x = 1.0/2.0*x1*x2*(1+g_xm)
    f2x = 1.0/2.0*x1*(1-x2)*(1+g_xm)
    f3x = 1.0/2.0*(1-x1)*(1+g_xm)

    return [f1x, f2x, f3x]

def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    nrun = args.nrun
    npilot = args.npilot
    TUNER_NAME = args.optimization

    tuning_metadata = {
        "tuning_problem_name": "DTLZ1",
        "machine_configuration": {
            "machine_name": "mymachine",
            "myprocessor": { "nodes": 1, "cores": 2}
        },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "mymachine": {
                "myprocessor": {
                    "nodes": 1,
                    "cores": 2
                }
            }
        }
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    problem = Categoricalnorm(["DTLZ1"], transform="onehot", name="problem")
    x1 = Real(0., 1., transform="normalize", name="x1")
    x2 = Real(0., 1., transform="normalize", name="x2")
    x3 = Real(0., 1., transform="normalize", name="x3")

    y1 = Real(float("-Inf"), float("Inf"), name="y1")
    y2 = Real(float("-Inf"), float("Inf"), name="y2")
    y3 = Real(float("-Inf"), float("Inf"), name="y3")

    input_space = Space([problem])
    parameter_space = Space([x1, x2, x3])
    output_space = Space([y1, y2, y3])
    constraints = {}
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)

    historydb = HistoryDB(meta_dict=tuning_metadata)

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    ## disable the following lines to use product of individual EIs as a single-valued acquisition function
    options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
    options['search_pop_size'] = 1000
    options['search_gen'] = 50
    options['search_more_samples'] = 5

    options['model_class'] = 'Model_LCM' #'Model_GPy_LCM'
    options['verbose'] = False
    options['sample_class'] = 'SampleOpenTURNS'

    options.validate(computer=computer)

    giventask = [["DTLZ1"]]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        (data, modeler, stats) = gt.MLA(NS=NS, Tgiven=giventask, NI=NI, NS1=npilot)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        import pymoo
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    problem:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            front = NonDominatedSorting(method="fast_non_dominated_sort").do(data.O[tid], only_non_dominated_front=True)
            # print('front id: ',front)
            fopts = data.O[tid][front]
            xopts = [data.P[tid][i] for i in front]
            print('    Popts ', xopts)
            print('    Oopts ', fopts.tolist())  

if __name__ == "__main__":
    main()
