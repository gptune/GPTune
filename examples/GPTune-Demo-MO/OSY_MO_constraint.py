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
import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all
import pickle

import argparse
from mpi4py import MPI
import numpy as np
import time

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')

    args = parser.parse_args()

    return args

def objectives(point):
    x1 = point["x1"]
    x2 = point["x2"]
    x3 = point["x3"]
    x4 = point["x4"]
    x5 = point["x5"]
    x6 = point["x6"]

    y1 = -1*(25*((x1-2)**2) + (x2-2)**2 + (x3-1)**2 + (x4-4)**2 + (x5-1)**2)
    y2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2

    return [y1, y2]

def cst1(x1, x2):
    return (x1+x2-2 >= 0) and (6-x1-x2 >=0) and (2-x2+x1 >= 0) and (2-x1+3*x2 >= 0)

def cst2(x3, x4):
    return 4 - (x3-3)**2 - x4 >= 0

def cst3(x5, x6):
    return (x5-3)**2 + x6 - 4 >= 0

def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    nrun = args.nrun
    npilot = args.npilot
    TUNER_NAME = args.optimization

    database_metadata = {
        "tuning_problem_name": "OSY_constraint",
        "machine_configuration": {
            "machine_name": "mymachine",
            "myprocessor": {
                "nodes": 1,
                "cores": 2
                }
            },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "mymachine": {
                "myprocessor": {
                    "nodes": 1,
                    "cores": 2
                    }
                }
            },
        "loadable_software_configurations": {}
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = database_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    problem = Categoricalnorm(["OSY"], transform="onehot", name="problem")
    x1 = Real(0., 10., transform="normalize", name="x1")
    x2 = Real(0., 10., transform="normalize", name="x2")
    x3 = Real(1., 5., transform="normalize", name="x3")
    x4 = Real(0., 6., transform="normalize", name="x4")
    x5 = Real(1., 5., transform="normalize", name="x5")
    x6 = Real(0., 10., transform="normalize", name="x6")
    y1 = Real(float("-Inf"), -150, name="y1", optimize=False)
    y2 = Real(float("-Inf"), float("Inf"), name="y2")

    input_space = Space([problem])
    parameter_space = Space([x1, x2, x3, x4, x5, x6])
    output_space = Space([y1, y2])
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)

    historydb = HistoryDB(meta_dict=database_metadata)

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
    options['search_gen'] = 10
    options['search_more_samples'] = 4

    # options['mpi_comm'] = None
    #options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_LCM_constrained' #'Model_LCM' #'Model_GPy_LCM'
    options['verbose'] = False
    # options['sample_algo'] = 'MCS'
    # options['sample_class'] = 'SampleLHSMDU'
    options['sample_class'] = 'SampleOpenTURNS'

    options.validate(computer=computer)

    giventask = [["OSY"]]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=npilot)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        import pygmo as pg
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    problem:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data.O[tid])
            front = ndf[0]
            # print('front id: ',front)
            fopts = data.O[tid][front]
            xopts = [data.P[tid][i] for i in front]
            print('    Popts ', xopts)
            print('    Oopts ', fopts.tolist())

    if True: # python plot
        from pymoo.factory import get_problem
        from pymoo.util.plotting import plot
        problem = get_problem("osy")
        pareto_front = problem.pareto_front()
        '''Plotting process'''
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        plt.plot(pf_X, pf_Y, color='magenta', label='Truth')

        PS = data.P[0]
        OS = data.O[0]

        y1 = []
        y2 = []
        for o in OS:
            y1.append(o[0])
            y2.append(o[1])

        plt.plot(y1, y2, 'o', color='black', label='Search')

        '''Pareto frontier selection process'''
        sorted_list = sorted([[y1[i], y2[i]] for i in range(len(y1))], reverse=False)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

        '''Plotting process'''
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        plt.plot(pf_X, pf_Y, color='red')

        #for i in range(len(OS)):
        #    label = PS[i]
        #    x = y1[i]
        #    y = y2[i]
        #    plt.annotate(label, # this is the text
        #                 (x,y), # this is the point to label
        #                 textcoords="offset points", # how to position the text
        #                 xytext=(0,10), # distance from text to points (x,y)
        #                 ha='center') # horizontal alignment can be left, right or center

        y1 = y1[npilot:nrun]
        y2 = y2[npilot:nrun]
        plt.plot(y1, y2, 'o', color='red', label='Search')

        plt.title("Tuning on OSY")
        plt.legend(loc="upper right")
        plt.xlabel('Y1')
        plt.ylabel('Y2')
        #plt.ylim([0,100])
        plt.show()

        #plt.figure()
        #Or = np.array(data.O[tid][front])
        #pickle.dump(Or, open(f"{options['search_algo']}_pareto.pkl", "wb"))

if __name__ == "__main__":
    main()
