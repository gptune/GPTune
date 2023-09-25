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
    # We use the mathematical model presented in the following paper:
    # Hamid Afshari, Warren Hare, Solomon Tesfamariam,
    # "Constrained multi-objective optimization algorithms: Review and Comparison with application in reinforced concrete structures",
    # Applied Soft Computing Journal 2019.

    Length = point["length"] # Length of beam
    Load = point["load"] # DL (dead load) + LL (live load)
    dbar = point["dbar"] # radius of steel bar

    x1 = point["x1"] # Cross-sectional area if steel reinforcement (A_s)
    x2 = point["x2"] # Effective depth of beam (d)
    x3 = point["x3"] # Beam width (b)
    x4 = point["x4"] # Compressive strength of concrete (f'_c)  that impacts the UC_s.

    G5 = 9*x2*x3*math.sqrt(x4)
    G4 = math.sqrt((x1*(200*x1+G5))/((x2**2)*(x3**2)*x4))
    G3 = (20*x1-1.41*x2*x3*math.sqrt(x4)*G4)**3
    G2 = (400*x1+G5-28.3*x2*x3*math.sqrt(x4)*G4)**2
    G1 = 10*x2 + 5*dbar + 602
    G_sub_1 = (x3*(G1**3))/12000 + (8000*G3-1200*x1*G2)/(2187*(x3**2)*(x4**1.5))
    G = (2844444*x1*G2 - 6320987*G3)/((x3**2)*(x4**1.5)) + ((x3**3)*(x4**1.5)*(G1**6)*(G_sub_1))/((9*(Length**6))*(Load**3))

    deflection = (5*(Length**4)*Load)/(G*math.sqrt(x4))
    cost = 1.5*(10**-3)*x1 + 1.2*(10**-5)*x2*x3 + 8.4*(10**-4)*x3

    return [deflection, cost]

def cst1(x1, x2, x3, x4):
    return 400*x1 - 14*x3*math.sqrt(x4) - 0.2*x2*x3*math.sqrt(x4) >= 0

def cst2(x1, x2, x3, x4):
    return x1/(x2*x3) - 0.0009*x4 <= 0

def cst3(x1, x2, x3, x4):
    return (x1*x2*x3*x4*(187-0.34*x4)-(57800*(x1**2)))/(0.55*x3*x4-0.001*x3*(x4**2)) >= 157500

def cst4(x2, x3):
    return 3*x3-x2 >= 70

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
        "tuning_problem_name": "RC_beam",
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

    # Length of beam
    length = Real(1, 100000, transform="normalize", name="length")
    # Load: DL (dead load) + LL (live load)
    load = Real(1, 1000, transform="normalize", name="load")
    # radius of steel bar
    dbar = Real(1, 1000, transform="normalize", name="dbar")

    # Cross-sectional area if steel reinforcement (A_s)
    x1 = Real(100, 500, transform="normalize", name="x1")
    # Effective depth of beam (d)
    x2 = Real(100, 500, transform="normalize", name="x2")
    # Beam width (b)
    x3 = Real(100, 500, transform="normalize", name="x3")
    # Compressive strength of concrete (f'_c)  that impacts the UC_s.
    x4 = Real(20, 40, transform="normalize", name="x4")

    deflection = Real(float("-Inf"), float("Inf"), name="deflection")
    cost = Real(float("-Inf"), float("Inf"), name="cost")

    input_space = Space([length, load, dbar])
    parameter_space = Space([x1, x2, x3, x4])
    output_space = Space([deflection, cost])
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3, "cst4": cst4}
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
    options['search_gen'] = 50
    options['search_more_samples'] = 5

    options['model_class'] = 'Model_LCM' #'Model_GPy_LCM'
    options['verbose'] = False
    options['sample_class'] = 'SampleOpenTURNS'

    options.validate(computer=computer)

    giventask = [[12000, 100, 100]]

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

    if True: # python plot
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

        #y1 = y1[npilot:nrun]
        #y2 = y2[npilot:nrun]
        #plt.plot(y1, y2, 'o', color='red', label='Search')

        plt.title("Tuning on RC Beam")
        plt.legend(loc="upper right")
        plt.xlabel('Deflection (mm)')
        plt.ylabel('Cost ($)')
        #plt.ylim([0,100])
        plt.show()

if __name__ == "__main__":
    main()
