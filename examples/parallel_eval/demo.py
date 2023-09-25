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
import logging
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from gptune import *

import argparse
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=100, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')
    parser.add_argument('-nbatch', type=int, default=0, help='Input task t value')

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

    import time
    time.sleep(5)

    print ("Function Evaluation: ", f)

    return [f]

def main():

    args = parse_args()
    tvalue = args.tvalue
    nrun = args.nrun
    npilot = args.npilot
    nbatch = args.nbatch

    tuning_problem_name = "Demo"+"-tvalue_"+str(tvalue)+"-npilot_"+str(npilot)+"-batch_num_"+str(nbatch)

    tuning_metadata = {
        "tuning_problem_name": tuning_problem_name,
        "sync_crowd_repo": "no",
        "no_load_check": "yes",
        "save_model": "yes"
    }

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)

    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=1, cores=8, hosts=None)
    data = Data(problem)
    options = Options()

    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_random_seed'] = nbatch
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = nbatch
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = nbatch

    options['distributed_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = True
    options['shared_memory_parallelism'] = True
    options['objective_multisample_threads'] = 4
    options['objective_multisample_processes'] = 1
    options['BO_objective_evaluation_parallelism'] = True

    options["oversubscribe"] = True

    options.validate(computer=computer)
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    giventask = [tvalue]
    (data, modeler, stats) = gt.SLA(NS=nrun, NS1=10, Tgiven=giventask)
    #giventask = [[tvalue], [tvalue+0.01]]
    #(data, modeler, stats) = gt.MLA(NS=nrun, NS1=10, NI=2, Tgiven=giventask)

    """ Print all input and parameter samples """
    print("stats: ", stats)
    print("    t: ", data.I)
    print("    Ps ", data.P)
    print("    Os ", data.O) #.tolist())
    print('    Popt ', data.P[np.argmin(data.O)], 'Oopt ', min(data.O)[0], 'nth ', np.argmin(data.O))

if __name__ == "__main__":
    main()

