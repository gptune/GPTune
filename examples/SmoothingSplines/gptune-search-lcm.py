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

mpirun -n 1 python ./gptune-search.py

"""


################################################################################
import sys
import os
import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import *

import argparse
from mpi4py import MPI
import numpy as np
import time

from callopentuner import OpenTuner
from callhpbandster import HpBandSter

import rpy2.robjects as robjects
import numpy as np
import math

def compute_rsquared(\
    baseline,\
    prediction,\
    total_points):

    mean_y = 0.0
    for i in range(total_points):
        mean_y += baseline[i]
    mean_y /= float(total_points)

    SS_res = 0.0
    for i in range(total_points):
        SS_res += math.pow((baseline[i] - prediction[i]), 2)

    SS_tot = 0.0
    for i in range(total_points):
        SS_tot += math.pow((baseline[i] - mean_y), 2)

    Rsquared = 1.0 - SS_res/SS_tot

    return Rsquared

def compute_arsquared(\
    baseline,\
    prediction,\
    total_points):

    mean_y = 0.0
    for i in range(total_points):
        mean_y += baseline[i]
    mean_y /= float(total_points)

    SS_res = 0.0
    for i in range(total_points):
        SS_res += math.pow((baseline[i] - prediction[i]), 2)

    SS_tot = 0.0
    for i in range(total_points):
        SS_tot += math.pow((baseline[i] - mean_y), 2)

    Rsquared = 1.0 - SS_res/SS_tot

    n = total_points
    p = 1
    Adjusted = 1.0 - (1.0-Rsquared)*float(n-1)/float(n-p-1)

    return Adjusted

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-task', type=str, default='1.0', help='Task')
    parser.add_argument('-size', type=str, default='10000', help='Number of dataset samples')
    parser.add_argument('-var', type=str, default='0.1', help='Variance of samples')

    args = parser.parse_args()

    return args

def objectives(point):

    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """

    t = point['t']
    k = int(point['k']) # nknots
    l = float(point['l']) # lambda

    with open("gptune-search-lcm.db/gptune-search."+str(nrun)+"."+task+"."+size+"."+var+".log", "a") as f_out:
        #f_out.write("NKnots,Lambda,RegressionTime,InTestTime,InMSE,InR2,InAR2,OutTestTime,OutMSE,OutR2,OutAR2\n")
        #lambda1 = float(l)/20.0
        lambda1 = l

        print ("K: "+str(k)+" L:"+str(lambda1))
        f_out.write(str(k)+","+str(lambda1)+",")

        r_y = robjects.FloatVector(Y_train)
        r_x = robjects.FloatVector(X_train)

        #try:
        t1 = time.time_ns()
        r_smooth_spline = robjects.r['smooth.spline']
        spline1 = r_smooth_spline(x=r_x, y=r_y, spar=lambda1, nknots=k)
        t2 = time.time_ns()

        r_time = (t2-t1)/1e9
        f_out.write(str(r_time)+",")

        t1 = time.time_ns()
        Y_train_spline = np.array(robjects.r['predict'](spline1,robjects.FloatVector(X_train)).rx2('y'))
        in_mse = np.mean((np.array(Y_train)-np.array(Y_train_spline))**2)
        in_R2 = compute_rsquared(Y_train, Y_train_spline, len(Y_train))
        in_AR2 = compute_arsquared(Y_train, Y_train_spline, len(Y_train))
        t2 = time.time_ns()
    
        in_test_time = (t2-t1)/1e9
    
        t1 = time.time_ns()
        Y_test_spline = np.array(robjects.r['predict'](spline1,robjects.FloatVector(X_test)).rx2('y'))
        out_mse = np.mean((np.array(Y_test)-np.array(Y_test_spline))**2)
        out_R2 = compute_rsquared(Y_test, Y_test_spline, len(Y_test))
        out_AR2 = compute_arsquared(Y_test, Y_test_spline, len(Y_test))
        t2 = time.time_ns()
    
        out_test_time = (t2-t1)/1e9
    
        f_out.write(str(in_test_time)+","+str(in_mse)+","+str(in_R2)+","+str(in_AR2)+","+str(out_test_time)+","+str(out_mse)+","+str(out_R2)+","+str(out_AR2)+"\n")

        return [in_mse]

def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    global nrun
    nrun = args.nrun
    global task
    task = args.task
    global var
    var = args.var
    global size
    size = args.size
    TUNER_NAME = args.optimization

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    t = Real(0., 10., transform="normalize", name="t")
    k = Integer(int(float(size)*0.1), int(float(size)*0.9), transform="normalize", name="k")
    l = Real(0., 1., transform="normalize", name="l")
    o = Real(float('-Inf'), float('Inf'), name="o")

    IS = Space([t])
    PS = Space([k, l])
    OS = Space([o])

    constraints = {}

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None)

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
    options['model_class'] = 'Model_LCM'
    #options['model_class'] = 'Model_GPy_LCM'
    options['verbose'] = False
    options.validate(computer=computer)

    giventask = [[float(task)]]

    global X_train
    global Y_train
    global X_test
    global Y_test

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    if not os.path.exists("gptune-search-lcm.db"):
        os.system("mkdir -p gptune-search-lcm.db")
    with open("gptune-search-lcm.db/gptune-search."+str(nrun)+"."+task+"."+size+"."+var+".log", "w") as f_out:
        f_out.write("NKnots,Lambda,RegressionTime,InTestTime,InMSE,InR2,InAR2,OutTestTime,OutMSE,OutR2,OutAR2\n")

    dataset = "gptune-demo-"+task+"-"+size+"-"+var
    trainset = dataset+"-train"
    with open("datagen/"+trainset, "r") as f_in:
        for dataline in f_in.readlines():
            data = dataline.split(",")
            X_train.append(float(data[0]))
            Y_train.append(float(data[1]))

    testset = dataset+"-test"
    with open("datagen/"+testset, "r") as f_in:
        for dataline in f_in.readlines():
            data = dataline.split(",")
            X_test.append(float(data[0]))
            Y_test.append(float(data[1]))

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))
        (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2))

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

if __name__ == "__main__":
    main()
