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
"""
Example of invocation of this script:

mpirun -n 1 python cnn_MB.py -ntrain 1000 -nvalid 200 -nprocmin_pernode 1 -ntask 20 -nrun 10
where:
    -ntrain/nvalid           number of training/validating data in CNN
    -nprocmin_pernode        minimum number of MPIs per node for launching the application code
    -ntask                   number of different tasks to be tuned
    -nrun                    number of calls per task
    
Description of the parameters of CNN-MNIST:
Task space:
    -ntrain/nvalid     number of training/validating data in CNN

Input space:
    lr:                learning rate
    optimizer:         optimizer of the CNN
    sgd_momentum:      the SGD momentum, only active if optimizer == SGD
    num_conv_layers:   number of convolution layers
    num_filters_1:     number of filters in the first conf layer
    num_filters_2:     number of filters in the second conf layer
    num_filters_3:     number of filters in the third conf layer
    dropout_rate:      dropout rate  
    num_fc_units:      number of hidden units in fully connected layer
    
"""

import sys, os
# add GPTunde path in front of all python pkg path
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../cnnMNIST-driver/"))
from cnnMNISTdriver import cnnMNISTdriver
import re
import numpy as np
import time
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter, HpBandSter_bandit
import math
import functools
import scipy


def objectives(point):
    bmin = point['bmin']
    bmax = point['bmax']
    eta = point['eta']   
    
    params = [(point["lr"], point["optimizer"], point["sgd_momentum"], 
               point["num_conv_layers"], point["num_filters_1"], 
               point["num_filters_2"], point["num_filters_3"], 
               point["dropout_rate"], point["num_fc_units"])]
    
    try:
        budget = int(point["budget"])
    except:
        budget = None
        
            
    validation_loss = cnnMNISTdriver(params, niter=1, 
                                  budget=budget, 
                                  max_epoch=bmax, batch_size=64,
                                  ntrain=point["ntrain"], nvalid=point["nvalid"])
    # print(params, ' valiation accuracy: ', accuracy)
    
    return validation_loss



def main(): 
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    # Parse command line arguments
    args = parse_args()
    bmin = args.bmin
    bmax = args.bmax
    eta = args.eta
    nrun = args.nrun
    npernode = args.npernode
    ntask = args.ntask
    Nloop = args.Nloop
    restart = args.restart
    TUNER_NAME = args.optimization
    TLA = False
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print(args)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    ntrain = Integer(1000, 10000, transform="normalize", name="ntrain")
    nvalid = Integer(256, 2048, transform="normalize", name="nvalid")
    
    lr = Real(1e-6, 1e-2, name="lr")
    optimizer = Categoricalnorm(['Adam', 'SGD'], transform="onehot", name="optimizer")
    sgd_momentum = Real(0, 0.99, name="sgd_momentum")
    num_conv_layers = Integer(1, 3, transform="normalize", name="num_conv_layers")
    num_filters_1 = Integer(4, 64, transform="normalize", name="num_filters_1")
    num_filters_2 = Integer(4, 64, transform="normalize", name="num_filters_2")
    num_filters_3 = Integer(4, 64, transform="normalize", name="num_filters_3")
    dropout_rate = Real(0, 0.9, name="dropout_rate")
    num_fc_units = Integer(8, 256, transform="normalize", name="num_fc_units")
    validation_loss = Real(float("-Inf"), float("Inf"), name="validation_loss")
    
    IS = Space([ntrain, nvalid])
    PS = Space([lr, optimizer, sgd_momentum, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units])
    OS = Space([validation_loss])
    
    constraints = {}
    constants={"nodes":nodes,"cores":cores,"npernode":npernode,"bmin":bmin,"bmax":bmax,"eta":eta}

    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, constants=constants) 
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['model_processes'] = 4 # parallel cholesky for each LCM kernel
    # options['model_threads'] = 1
    
    # options['model_restarts'] = args.Nrestarts
    # options['distributed_memory_parallelism'] = False
    
    # parallel model restart
    options['model_restarts'] = restart
    options['distributed_memory_parallelism'] = False
    
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class'] = 'Model_LCM' # Model_GPy_LCM or Model_LCM
    options['verbose'] = False
    
    
    options['budget_min'] = bmin
    options['budget_max'] = bmax
    options['budget_base'] = eta
    smax = int(np.floor(np.log(options['budget_max']/options['budget_min'])/np.log(options['budget_base'])))
    budgets = [options['budget_max'] /options['budget_base']**x for x in range(smax+1)]
    NSs = [int((smax+1)/(s+1))*options['budget_base']**s for s in range(smax+1)] 
    NSs_all = NSs.copy()
    budget_all = budgets.copy()
    for s in range(smax+1):
        for n in range(s):
            NSs_all.append(int(NSs[s]/options['budget_base']**(n+1)))
            budget_all.append(int(budgets[s]*options['budget_base']**(n+1)))
    Ntotal = int(sum(NSs_all) * Nloop)
    Btotal = int(np.dot(np.array(NSs_all), np.array(budget_all))/options['budget_max'] * Nloop) # total number of evaluations at highest budget -- used for single-fidelity tuners
    print(f"bmin = {bmin}, bmax = {bmax}, eta = {eta}, smax = {smax}")
    print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)
    print("total number of samples: ", Ntotal)
    print("total number of evaluations at highest budget: ", Btotal)
    print()
    
    options.validate(computer = computer)
    
    data = Data(problem)
    # giventask = [[0.2, 0.5]]
    
    if ntask == 1:
        # giventask = [[args.ntrain, args.nvalid]]
        giventask = [[3000, 1000]]
    
    
    NI=len(giventask)
    assert NI == ntask # make sure number of tasks match

    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NS = Btotal
        if args.nrun > 0:
            NS = args.nrun
        NS1 = max(NS//2, 1)
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS1)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [ntrain, nvalid] = [{data.I[tid][0]}, {data.I[tid][1]}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    
    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [ntrain, nvalid] = [{data.I[tid][0]}, {data.I[tid][1]}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid][:NS])], 'Oopt ', min(data.O[tid][:NS])[0], 'nth ', np.argmin(data.O[tid][:NS]))

    # single-fidelity version of hpbandster
    if(TUNER_NAME=='TPE'):
        NS = Btotal
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [ntrain, nvalid] = [{data.I[tid][0]}, {data.I[tid][1]}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
    
    if(TUNER_NAME=='GPTuneBand'):
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
        (data, stats, data_hist)=gt.MB_LCM(NS = Nloop, Igiven = giventask)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [ntrain, nvalid] = [{data.I[tid][0]}, {data.I[tid][1]}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            nth = np.argmin(data.O[tid])
            Popt = data.P[tid][nth]
            # find which arm and which sample the optimal param is from
            for arm in range(len(data_hist.P)):
                try:
                    idx = (data_hist.P[arm]).index(Popt)
                    arm_opt = arm
                except ValueError:
                    pass
            print('    Popt ', Popt, 'Oopt ', min(data.O[tid])[0], 'nth ', nth, 'nth-bandit (s, nth) = ', (arm_opt, idx))
               
         
    # multi-fidelity version                
    if(TUNER_NAME=='hpbandster'):
        NS = Ntotal
        (data,stats)=HpBandSter_bandit(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [ntrain, nvalid] = [{data.I[tid][0]}, {data.I[tid][1]}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            # print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            max_budget = 0.
            Oopt = 99999
            Popt = None
            nth = None
            for idx, (config, out) in enumerate(zip(data.P[tid], data.O[tid].tolist())):
                for subout in out[0]:
                    budget_cur = subout[0]
                    if budget_cur > max_budget:
                        max_budget = budget_cur
                        Oopt = subout[1]
                        Popt = config
                        nth = idx
                    elif budget_cur == max_budget:
                        if subout[1] < Oopt:
                            Oopt = subout[1]
                            Popt = config
                            nth = idx                    
            print('    Popt ', Popt, 'Oopt ', Oopt, 'nth ', nth)

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments

    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-npernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-ntrain', type=int, default=3000, help='Number of training data')
    parser.add_argument('-nvalid', type=int, default=1000, help='Number of testing')
    parser.add_argument('-nrun', type=int, default=2, help='Number of runs per task')
    parser.add_argument('-bmin', type=int, default=1,  help='minimum fidelity for a bandit structure')
    parser.add_argument('-bmax', type=int, default=8, help='maximum fidelity for a bandit structure')
    parser.add_argument('-eta', type=int, default=2, help='base value for a bandit structure')
    parser.add_argument('-Nloop', type=int, default=1, help='number of GPTuneBand loops')
    parser.add_argument('-restart', type=int, default=1, help='number of GPTune MLA restart')


    args   = parser.parse_args()
    return args


if __name__ == "__main__":
 
    main()
