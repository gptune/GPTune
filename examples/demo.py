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
import sys
import os
import mpi4py
from mpi4py import MPI
import numpy as np
import time
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import logging

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
    f = d * e
    # print('dd',test)

    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f*(1+np.random.uniform()*0.1)]


""" Plot the objective function for t=1,2,3,4,5,6 """
def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="offset points",arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(210,5), **kw)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="time")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    # problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, models)  # with performance model
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)  # no performance model

    computer = Computer(nodes=1, cores=16, hosts=None)
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

    os.environ['TUNER_NAME'] = 'hpbandster'
    
    giventask = [[6]]
    # giventask = [[i] for i in np.arange(0, 10, 0.5).tolist()]

    NI=len(giventask)
    NS=100	    
    
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
            print("    t:%d " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
        
    

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%d " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%d " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))




    plot=0
    if plot==1:
        x = np.arange(0., 1., 0.00001)
        Nplot=9.5
        for t in np.linspace(0,Nplot,20):
            fig = plt.figure(figsize=[12.8, 9.6])
            I_orig=[t]
            kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}

            y=np.zeros([len(x),1])
            for i in range(len(x)):
                P_orig=[x[i]]
                kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                kwargs.update(kwargst)
                y[i]=objectives(kwargs) 
            fontsize=30
            plt.rcParams.update({'font.size': 21})
            plt.plot(x, y, 'b')
            plt.xlabel('x',fontsize=fontsize+2)
            plt.ylabel('y(t,x)',fontsize=fontsize+2)
            plt.title('t=%d'%t,fontsize=fontsize+2)
            print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())    
        
            annot_min(x,y)
            # plt.show()
            # plt.show(block=False)
            fig.savefig('obj_t_%d.eps'%t)                

