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
import json

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

    os.environ['TUNER_NAME'] = 'GPTune'

    giventask = [[6]]
    # giventask = [[i] for i in np.arange(0, 10, 0.5).tolist()]

    NI=len(giventask)
    NS=100

    # Load data from JSON
    try:
        with open("demo.json", "r") as f_in:
            data = Data(problem) # initial data

            history_data = json.load(f_in)

            num_tasks = len(history_data["perf_data"])
            IS_history = []
            for t in range(num_tasks):
                input_dict = history_data["perf_data"][t]["I"]
                IS_history.append(np.array([input_dict[input_space[k].name] for k in range(len(input_space))]))
            data.I = IS_history

            PS_history = []
            OS_history = []
            for t in range(num_tasks):
                PS_history_t = []
                OS_history_t = []
                num_evals = len(history_data["perf_data"][t]["func_eval"])
                for i in range(num_evals):
                    func_eval = history_data["perf_data"][t]["func_eval"][i]
                    PS_history_t.append([func_eval["P"][parameter_space[k].name] for k in range(len(parameter_space))])
                    OS_history_t.append([func_eval["O"][output_space[k].name] for k in range(len(output_space))])

                PS_history.append(PS_history_t)
                OS_history.append(OS_history_t)
            print (PS_history)
            print (OS_history)
            data.P = PS_history
            data.O = np.array(OS_history)

    except (OSError, IOError) as e:
        print ("no previous evaluation")
        data = Data(problem)

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        #data = Data(problem)
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

    # Save data into JSON
    with open("demo.json", "w") as f_out:
        json_data = {}
        json_data["id"] = 0
        json_data["name"] = "demo"
        json_data["perf_data"] = []
        num_tasks = len(data.I)
        for t in range(num_tasks):
            num_runs = len(data.P[t])
            run_data = []
            for i in range(num_runs):
                run_data.append({
                    "id":i,
                    "P":{parameter_space[k].name:data.P[t][i].tolist()[k] for k in range(len(data.P[t][i].tolist()))},
                    "O":{output_space[k].name:data.O[t][i].tolist()[k] for k in range(len(data.O[t][i].tolist()))}
                    })

            I_list = np.array(data.I[t]).tolist()
            json_data["perf_data"].append({
                    "id":t,
                    "I":{input_space[k].name:I_list[k] for k in range(len(I_list))},
                    "func_eval":run_data
                })

        json.dump(json_data, f_out, indent=4)

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

