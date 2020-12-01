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

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import argparse
import sys
import os
import mpi4py
from mpi4py import MPI
import numpy as np
from numpy import *
import time
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import logging
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-plot', type=int, default=0, help='Whether to plot the objective function')
    parser.add_argument('-nrep', type=int, default=1, help='Number of times to repeat a tuning exp')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')


    args = parser.parse_args()

    return args
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


# test=1  # make sure to set global variables here, rather than in the main function 
def models(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    # global test
    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1
    # print('dd',test)

    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f*(1+np.random.uniform()*0.1)]



def main():
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()

    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    perfmodel = args.perfmodel
    plot = args.plot    
    nrep = args.nrep    

    os.environ['MACHINE_NAME'] = machine   
    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="time")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    if(perfmodel==1):    
        problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, models)  # with performance model
    else:    
        problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)  # no performance model

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


    # options['mpi_comm'] = None
    #options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_LCM' #'Model_GPy_LCM'
    options['verbose'] = False
    # options['sample_algo'] = 'MCS'
    # options['sample_class'] = 'SampleLHSMDU'

    options.validate(computer=computer)

    allavrs=[]
    allmaxs=[]
    allmins=[]
    times=[]

    # os.environ['TUNER_NAME'] = 'hpbandster' #'hpbandster'
    
    giventask = [[6]]
    # giventask = [[i] for i in np.arange(0, 10, 0.5).tolist()]

    NI=len(giventask)
    # NS=80	    
    NREP=nrep
    NSS=[10, 20, 40]

    for TUNER_NAME in ['GPTune','hpbandster','opentuner']:
        t1 = time.time_ns()
        
        mins=np.zeros(len(NSS))
        maxs=np.zeros(len(NSS))
        avr=np.zeros(len(NSS))  
        for ss in range(len(NSS)):
            NS=NSS[ss]
            opts=np.zeros(NREP)
            for ii in range(NREP):
                if(TUNER_NAME=='GPTune'):
                    data = Data(problem)
                    gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))
                    (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2))
                    # (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS-1)
                    print("stats: ", stats)
                    """ Print all input and parameter samples """
                    for tid in range(NI):
                        print("tid: %d" % (tid))
                        print("    t:%f " % (data.I[tid][0]))
                        print("    Ps ", data.P[tid])
                        print("    Os ", data.O[tid].tolist())
                        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
                    
                

                if(TUNER_NAME=='opentuner'):
                    (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
                    print("stats: ", stats)
                    """ Print all input and parameter samples """
                    for tid in range(NI):
                        print("tid: %d" % (tid))
                        print("    t:%f " % (data.I[tid][0]))
                        print("    Ps ", data.P[tid])
                        print("    Os ", data.O[tid].tolist())
                        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

                if(TUNER_NAME=='hpbandster'):
                    (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
                    print("stats: ", stats)
                    """ Print all input and parameter samples """
                    for tid in range(NI):
                        print("tid: %d" % (tid))
                        print("    t:%f " % (data.I[tid][0]))
                        print("    Ps ", data.P[tid])
                        print("    Os ", data.O[tid].tolist())
                        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

                opts[ii] = min(data.O[0])[0]
            avr[ss]=np.average(opts)
            maxs[ss]=np.max(opts)
            mins[ss]=np.min(opts)
                
        allavrs.append(avr)
        allmaxs.append(maxs)
        allmins.append(mins)
        
        t2 = time.time_ns()
        times.append((t2-t1)/1e9)

    print(allavrs)
    print(allmaxs)
    print(allmins)
    print(times)


    ####  t=1
    # allavrs=[array([0.8927, 0.8198, 0.7856, 0.768 , 0.7442]), array([0.8636, 0.866 , 0.7654, 0.7648, 0.7586]), array([0.8423, 0.7759, 0.7606, 0.7498, 0.741 ])]
    # allmaxs=[array([0.9681, 0.959 , 0.8696, 0.8151, 0.7681]), array([0.9569, 0.9592, 0.8267, 0.8083, 0.8147]), array([0.9428, 0.8278, 0.8411, 0.7722, 0.7531])]
    # allmins=[array([0.7688, 0.7356, 0.7411, 0.7359, 0.7355]), array([0.7681, 0.7354, 0.7356, 0.7355, 0.7354]), array([0.7694, 0.7383, 0.7356, 0.7354, 0.7354])]

    ####  t=3
    # allavrs=[array([0.9407, 0.9352, 0.8292, 0.7633, 0.7069]), array([0.8918, 0.9355, 0.8386, 0.7209, 0.6713]), array([0.8748, 0.8151, 0.7458, 0.6917, 0.6436])]
    # allmaxs=[array([0.9999, 0.9943, 0.9761, 0.8472, 0.7827]), array([0.9974, 0.9985, 0.972 , 0.8739, 0.7477]), array([1.    , 0.9774, 0.9771, 0.9705, 0.7227])]
    # allmins=[array([0.7123, 0.7474, 0.626 , 0.6202, 0.6101]), array([0.7045, 0.6145, 0.647 , 0.6101, 0.6101]), array([0.6307, 0.6203, 0.6102, 0.6102, 0.6101])]

    ####  t=4
    # allavrs=[array([0.9093, 0.9388, 0.8455, 0.7118, 0.6694]), array([0.8675, 0.7928, 0.7793, 0.7134, 0.7454]), array([0.8656, 0.8245, 0.707 , 0.685 , 0.6541])]
    # allmaxs=[array([1.    , 1.    , 0.9271, 0.8848, 0.7925]), array([1.    , 0.9927, 0.9877, 0.9964, 0.9606]), array([0.9964, 0.9995, 0.9905, 0.9834, 0.7493])]
    # allmins=[array([0.6321, 0.7726, 0.6383, 0.6006, 0.5905]), array([0.64  , 0.6002, 0.6539, 0.5902, 0.5909]), array([0.6404, 0.6015, 0.5935, 0.5902, 0.5902])]

    ####  t=6
    # allavrs=[array([0.9641, 0.9199, 0.7434, 0.7368, 0.696 ]), array([0.9214, 0.8602, 0.8214, 0.7431, 0.6681]), array([0.9394, 0.8431, 0.8218, 0.6557, 0.6129])]
    # allmaxs=[array([1.    , 0.9998, 0.9344, 0.9853, 0.771 ]), array([1.    , 1.    , 0.9994, 0.9883, 0.8733]), array([1.    , 0.9999, 0.9896, 0.7418, 0.6798])]
    # allmins=[array([0.7417, 0.6782, 0.627 , 0.5558, 0.6149]), array([0.6552, 0.6425, 0.5887, 0.5657, 0.5109]), array([0.5947, 0.6055, 0.7051, 0.5272, 0.5109])]

    if(plot==1):
        fontsize=24
        fig = plt.figure(figsize=[12.8, 9.6])

        plt.rcParams.update({'font.size': fontsize})

        plt.errorbar(np.array(NSS)-0.3, allavrs[0], yerr=[allavrs[0]-allmins[0],allmaxs[0]-allavrs[0]], capsize=10, elinewidth=2, markeredgewidth=5, fmt='o', label='GPTune')
        plt.errorbar(np.array(NSS), allavrs[1], yerr=[allavrs[1]-allmins[1],allmaxs[1]-allavrs[1]], capsize=10, elinewidth=2, markeredgewidth=5, fmt='o', label='hpbandster')
        plt.errorbar(np.array(NSS)+0.3, allavrs[2], yerr=[allavrs[2]-allmins[2],allmaxs[2]-allavrs[2]], capsize=10, elinewidth=2, markeredgewidth=5, fmt='o', label='opentuner')
        plt.plot([NSS[0]-5,NSS[-1]+5], [0.510885, 0.510885], c='black', linestyle=':')  # t=6
        # plt.plot([NSS[0]-5,NSS[-1]+5], [0.735, 0.735], c='black', linestyle=':')  # t=1
        # plt.plot([NSS[0]-5,NSS[-1]+5], [0.61012, 0.61012], c='black', linestyle=':') # t=3
        # plt.plot([NSS[0]-5,NSS[-1]+5], [0.59020, 0.59020], c='black', linestyle=':') # t=4

        plt.xlabel('NS',fontsize=fontsize+2)
        plt.ylabel('f_min',fontsize=fontsize+2)
        plt.legend(loc='upper right')
        plt.show(block=False)
        plt.pause(0.5)
        input("Press [enter] to continue.")    
        fig.savefig('fmins_t%d.eps'%int(giventask[0][0]))     




if __name__ == "__main__":
    main()