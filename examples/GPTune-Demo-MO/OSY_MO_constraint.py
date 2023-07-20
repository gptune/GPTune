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
    parser.add_argument('-constraint_mode', type=int, default=1, help='Number of initial runs per task')

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

    constraint_mode = args.constraint_mode

    database_metadata = {
        "tuning_problem_name": "OSY_constraint_"+str(constraint_mode),
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

    if constraint_mode == 1:
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

        options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
        options['search_pop_size'] = 1000
        options['search_gen'] = 50
        options['search_more_samples'] = 4
        options['model_class'] = 'Model_GPy_LCM'
        options['model_output_constraint'] = "LargeNum" #True
        options['model_bigval_LargeNum'] = 100
        options['verbose'] = False
        options['sample_class'] = 'SampleOpenTURNS'
        options.validate(computer=computer)

    elif constraint_mode == 2:
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

        options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
        options['search_pop_size'] = 1000
        options['search_gen'] = 50
        options['search_more_samples'] = 4
        options['model_class'] = 'Model_GPy_LCM'
        options['model_output_constraint'] = "Ignore"
        options['verbose'] = False
        options['sample_class'] = 'SampleOpenTURNS'
        options.validate(computer=computer)

    elif constraint_mode == 3:
        problem = Categoricalnorm(["OSY"], transform="onehot", name="problem")
        x1 = Real(0., 10., transform="normalize", name="x1")
        x2 = Real(0., 10., transform="normalize", name="x2")
        x3 = Real(1., 5., transform="normalize", name="x3")
        x4 = Real(0., 6., transform="normalize", name="x4")
        x5 = Real(1., 5., transform="normalize", name="x5")
        x6 = Real(0., 10., transform="normalize", name="x6")
        y1 = Real(float("-Inf"), -150, name="y1")
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

        options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
        options['search_pop_size'] = 1000
        options['search_gen'] = 50

        options['search_more_samples'] = 4
        options['model_class'] = 'Model_GPy_LCM'
        options['model_output_constraint'] = "LargeNum"
        options['model_bigval_LargeNum'] = 100
        options['verbose'] = False
        options['sample_class'] = 'SampleOpenTURNS'
        options.validate(computer=computer)

    elif constraint_mode == 4:
        problem = Categoricalnorm(["OSY"], transform="onehot", name="problem")
        x1 = Real(0., 10., transform="normalize", name="x1")
        x2 = Real(0., 10., transform="normalize", name="x2")
        x3 = Real(1., 5., transform="normalize", name="x3")
        x4 = Real(0., 6., transform="normalize", name="x4")
        x5 = Real(1., 5., transform="normalize", name="x5")
        x6 = Real(0., 10., transform="normalize", name="x6")
        y1 = Real(float("-Inf"), -150, name="y1")
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

        options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
        options['search_pop_size'] = 1000
        options['search_gen'] = 50

        options['search_more_samples'] = 4
        options['model_class'] = 'Model_GPy_LCM'
        options['model_output_constraint'] = "Ignore"
        options['verbose'] = False
        options['sample_class'] = 'SampleOpenTURNS'
        options.validate(computer=computer)

    giventask = [["OSY"]]

    NI=len(giventask)
    NS=nrun

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

        plt.plot(y1, y2, 'o', color='black', label='Search (initial random samples)')

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

        for i in range(len(OS)):
            label = i+1
            #label = PS[i]
            x = y1[i]
            y = y2[i]
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center

        y1 = y1[npilot:nrun]
        y2 = y2[npilot:nrun]
        plt.plot(y1, y2, 'o', color='red', label='Search (samples from BO)')

        if constraint_mode == 1:
            plt.title("Tuning on OSY \n (Output constraint: Y1 < -150; Minimize Y2 only) \n (Option: Large value)")
        elif constraint_mode == 2:
            plt.title("Tuning on OSY \n (Output constraint: Y1 < -150; Minimize Y2 only) \n (Option: Ignore)")
        elif constraint_mode == 3:
            plt.title("Tuning on OSY \n (Output constraint: Y1 < -150; Minimize Y1 and Y2) \n (Option: Large value)")
        elif constraint_mode == 4:
            plt.title("Tuning on OSY \n (Output constraint: Y1 < -150; Minimize Y1 and Y2) \n (Option: Ignore)")

        plt.legend(loc="upper right")
        plt.xlabel('Y1')
        plt.ylabel('Y2')
        #plt.ylim([0,100])
        #plt.show()
        plt.savefig("OSY_MO_Constraint_Mode"+str(constraint_mode)+".pdf")

        #plt.figure()
        #Or = np.array(data.O[tid][front])
        #pickle.dump(Or, open(f"{options['search_algo']}_pareto.pkl", "wb"))

if __name__ == "__main__":
    main()
