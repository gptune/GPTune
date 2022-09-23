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
import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from gptune import *

import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=100, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=1, help='Number of initial runs per task')
    parser.add_argument('-tvalue', type=float, default=1.2, help='Input task t value')

    args = parser.parse_args()

    return args

def objectives(point):
    global model_function

    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    t = point['t']

    if t == 1.0:
        point  = { "t": 1.0, "x": point['x'] }
        ret = model_function(point)
        print (ret)
        return ret["y"]
    else:
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

def LoadModels():

    global model_function

    with open("gptune.db/GPTune-Demo-1.0.json","r") as f_in:
        print (f_in)
        function_evaluations = json.load(f_in)["func_eval"]
        print (function_evaluations)

        x_observed = []
        y_observed = []
        for func_eval in function_evaluations:
            x_observed.append(func_eval["tuning_parameter"]["x"])
            y_observed.append(func_eval["evaluation_result"]["y"])

        metadata = {
            "tuning_problem_name":"GPTune-Demo-1.0",
            "modeler":"Model_GPy_LCM",
            "task_parameter":[[1.0]],
            "input_space": [{"name":"t","type":"real","transformer":"normalize","lower_bound":0.0,"upper_bound":10.0}],
            "parameter_space": [{"name":"x","type":"real","transformer":"normalize","lower_bound":0.0,"upper_bound":1.0}],
            "output_space": [{"name":"y","type":"real","transformer":"identity","lower_bound":float("-Inf"),"upper_bound":float("Inf")}]
        }

        model_function = BuildSurrogateModel(metadata = metadata, function_evaluations = function_evaluations)

    return

def main():

    global model_function
    LoadModels()

    args = parse_args()
    tvalue = args.tvalue
    nrun = args.nrun
    npilot = args.npilot

    tuning_metadata = {
        "tuning_problem_name": "GPTune-Demo-TLA-LCM-"+str(tvalue),
        "sync_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "Cori",
            "haswell": { "nodes": 1, "cores": 32 }
        },
        "software_configuration": {}
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)
    historydb = HistoryDB(meta_dict=tuning_metadata)
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
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)

    giventask = [[1.0],[tvalue]]
    NI=len(giventask)
    NS=nrun

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    #(data, modeler, stats) = gt.TLA3(NS=NS, Tgiven=giventask, NI=NI, NS1=int(NS/2), models_transfer = LoadModels())
    #(data, modeler, stats) = gt.TLA3(NS=NS, Tgiven=giventask, NI=NI, NS1=int(NS/2), models_transfer = None)
    (data, modeler, stats) = gt.MLA(NS=NS, Tgiven=giventask, NI=NI, NS1=npilot)
    # (data, modeler, stats) = gt.MLA(NS=NS, Tgiven=giventask, NI=NI, NS1=NS-1)
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
