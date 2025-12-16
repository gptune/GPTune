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


"""
Example of invocation of this script:

mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

where:
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -perfmodel is whether a coarse performance model is used
    -optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""


################################################################################
import sys
import os
# import mpi4py
import logging
import pandas as pd
df = pd.read_csv('./output-jobs-and-subjobs-301-304.csv')


logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from GPTune.gptune import * # import all


import argparse
# from mpi4py import MPI
import numpy as np
import time

from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter



# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')

    args = parser.parse_args()

    return args

def objectives(point):
    mc, kc = point['mc'], point['kc']#, point['nc']

    # Round mc, nc, kc to the nearest multiple of 8
    mc = max(np.round(mc / 12) * 12, 12)
    kc = max(np.round(kc / 16) * 16, 16)
    nc = 4080#np.round(nc / 6) * 6

    # Find the row in the DataFrame where mc, nc, kc match
    row = df[(df['mc'] == mc) & (df['nc'] == nc) & (df['kc'] == kc)]
    
    # If there's no exact match, return 1
    if not row.empty:
        #return np.array([row['time'].values[0]])
        return [row['min_jobs(min_subjobs(min_sec)'].values[0]]
        #return np.array([row['min_jobs(min_subjobs(min_sec)'].values[0]])
    else:
        return [1.]#np.array([1.])  # Return 1 if the combination is not found



def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    nrun = args.nrun
    TUNER_NAME = args.optimization

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    m = Integer(1000, 10000, transform="normalize", name="m")
    n = Integer(1000, 10000, transform="normalize", name="n")
    k = Integer(1000, 10000, transform="normalize", name="k")


    mc = Integer(1, 2004, transform="normalize", name="mc")
    kc = Integer(1, 2160, transform="normalize", name="kc")

    input_space = Space([m,n,k])
    parameter_space = Space([mc,kc])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {}
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)  # no performance model

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    # options['objective_evaluation_parallelism'] = True
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 4
    # options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    options['model_optimzier'] = 'lbfgs'

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    # Use the following two lines if you want to specify a certain random seed for the random pilot sampling
    # options['sample_algo'] = 'MCS'
    options['sample_class'] = 'SampleLHSMDU'
    options['sample_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for surrogate modeling
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for the search phase
    # options['search_class'] = 'SearchSciPy'
    options['search_random_seed'] = 0

    # options['search_class'] = 'SearchSciPy'
    # options['search_algo'] = 'l-bfgs-b'

    # options['search_more_samples'] = 4
    # options['search_af']='EI'
    # options['search_pop_size']=1000
    # options['search_ucb_beta']=0.01

    options['verbose'] = False
    options.validate(computer=computer)

    print(options)

    giventask = [[2000,2000,2000]]

    NI=len(giventask)
    NS=nrun
    NS1=10

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)

        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
        # gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__), historydb=False) ## Run GPTune without database

        (data, modeler, stats) = gt.MLA(NS=NS, NS1=NS1, NI=NI, Tgiven=giventask)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d k:%d " % (data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d k:%d " % (data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d k:%d " % (data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from GPTune.callcgp import cGP
        options['EXAMPLE_NAME_CGP']='BLISv2'
        options['N_PILOT_CGP']=NS1
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        options['METHOD_CGP']  = 'FREQUENTIST'
        options['N_BURNIN_CGP'] = 500
        options['N_MCMCSAMPLES_CGP'] = 500
        options['N_INFERENCE_CGP'] = 300
        options['EXPLORATION_RATE_CGP'] = 0.8
        options['NO_CLUSTER_CGP'] = False
        options['N_NEIGHBORS_CGP'] = 2
        options['N_COMPONENTS_CGP'] = 3

        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d k:%d " % (data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()
