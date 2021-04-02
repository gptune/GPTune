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
    -plot is whether the objective function and the model will be plotted
"""

################################################################################
import sys
import os
import mpi4py
import logging
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True
import matplotlib.pyplot as plt

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import argparse
from mpi4py import MPI
import numpy as np
import time
from callopentuner import OpenTuner
from callhpbandster import HpBandSter



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
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-plot', type=int, default=0, help='Whether to plot the objective function')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')


    args = parser.parse_args()

    return args


""" Plot the objective function for t=1,2,3,4,5,6 """
def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="offset points",arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top", fontsize=30)
    ax.annotate(text, xy=(xmin, ymin), xytext=(300,10), **kw)


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
    print('t',t,'x',x,'f',f)
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


def predict_aug(modeler, gt, point,tid):   # point is the orginal space
    x =point['x']
    xNorm = gt.problem.PS.transform([[x]])
    xi0 = gt.problem.PS.inverse_transform(np.array(xNorm, ndmin=2))
    xi=xi0[0]

    IOrig = gt.data.I[tid]

    # point0 = gt.data.D
    point2 = {gt.problem.IS[k].name: IOrig[k] for k in range(gt.problem.DI)}
    point  = {gt.problem.PS[k].name: xi[k] for k in range(gt.problem.DP)}
    # point.update(point0)
    point.update(point2)
    # print("point", point)

    xNorm = gt.problem.PS.transform(xi0)[0]
    if(gt.problem.models is not None):
        if(gt.problem.driverabspath is not None):
            modulename = Path(gt.problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(gt.problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module
        else:
            raise Exception('performance models require passing driverabspath to GPTune')
        # modeldata= self.problem.models(point)
        modeldata= module.models(point)
        xNorm = np.hstack((xNorm,modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
        # print(xNorm)
    (mu, var) = modeler[0].predict(xNorm, tid=tid)
    return (mu, var)



def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()

    ntask = args.ntask
    nrun = args.nrun
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel
    plot = args.plot
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), transform="normalize", name="y")])
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

    # options['model_processes'] = 1
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


    # giventask = [[6]]
    giventask = [[i] for i in np.arange(0, ntask/2, 0.5).tolist()]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']

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


    if plot==1:
        # fig = plt.figure(figsize=[12.8, 9.6])
        x = np.arange(0., 1., 0.0001)
        for tid in range(len(data.I)):
            fig = plt.figure(figsize=[12.8, 9.6])
            p = data.I[tid]
            t = p[0]
            I_orig=p
            kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}
            y=np.zeros([len(x),1])
            y_mean=np.zeros([len(x)])
            y_std=np.zeros([len(x)])
            for i in range(len(x)):
                P_orig=[x[i]]
                kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                kwargs.update(kwargst)
                y[i]=objectives(kwargs)
                if(TUNER_NAME=='GPTune'):
                    (y_mean[i],var) = predict_aug(modeler, gt, kwargs,tid)
                    y_std[i]=np.sqrt(var)
                    # print(y_mean[i],y_std[i],y[i])
            fontsize=40
            plt.rcParams.update({'font.size': 40})
            plt.plot(x, y, 'b',lw=2,label='true')

            plt.plot(x, y_mean, 'k', lw=3, zorder=9, label='prediction')
            plt.fill_between(x, y_mean - y_std, y_mean + y_std,alpha=0.2, color='k')
            # print(data.P[tid])
            plt.scatter(data.P[tid], data.O[tid], c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label='sample')

            plt.xlabel('x',fontsize=fontsize+2)
            plt.ylabel('y(t,x)',fontsize=fontsize+2)
            plt.title('t=%f'%t,fontsize=fontsize+2)
            print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())
            # legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            # legend = plt.legend(loc='upper right', shadow=False, fontsize=fontsize)
            annot_min(x,y)
            # plt.show()
            plt.show(block=False)
            plt.pause(0.5)
            input("Press [enter] to continue.")
            fig.savefig('obj_t_%f.pdf'%t)


if __name__ == "__main__":
    main()
