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
sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
api_key = os.getenv("CROWDTUNING_API_KEY")

from gptune import *

import argparse
import numpy as np

import crowdtune

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=100, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')
    parser.add_argument('-tid_source', type=int, default=0, help='Source task ID')
    parser.add_argument('-tid_target', type=int, default=0, help='Target task ID')
    parser.add_argument('-nbatch', type=int, default=0, help='Input task t value')
    parser.add_argument('-tuning_method', type=str, default='SLA', help='Tuning method')

    args = parser.parse_args()

    return args

def LoadSourceFunctionEvaluations(tid_source=0):

    with open("tasklist.json", "r") as f_in:
        source_task = json.load(f_in)["source_tasks"][tid_source]
        a_value = source_task["a"]
        b_value = source_task["b"]
        c_value = source_task["c"]
        r_value = source_task["r"]
        s_value = source_task["s"]
        t_value = source_task["t"]

    problem_space = {
        "input_space": [
            {"name":"a", "value": a_value},
            {"name":"b", "value": b_value},
            {"name":"c", "value": c_value},
            {"name":"r", "value": r_value},
            {"name":"s", "value": s_value},
            {"name":"t", "value": t_value}
        ],
        "constants": [],
        "parameter_space": [
            {"name":"x1", "type":"real", "transformer":"normalize", "lower_bound":-5.0, "upper_bound":10.0},
            {"name":"x2", "type":"real", "transformer":"normalize", "lower_bound":0.0, "upper_bound":15.0}
        ],
        "output_space": [
            {"name":"y", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("Inf")}
        ]
    }

    configuration_space = {}

    ret = crowdtune.QueryFunctionEvaluations(api_key = api_key,
            tuning_problem_name = "Branin",
            problem_space = problem_space,
            configuration_space = configuration_space)

    print ("crowdtuning API, downloaded function evaluations: ", ret)

    return [ret]

def objectives(point):

    x1 = point['x1']
    x2 = point['x2']
    a = point['a']
    b = point['b']
    c = point['c']
    r = point['r']
    s = point['s']
    t = point['t']

    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    #y = y.reshape(-1, 1)
    #y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)

    return [y]

def main():

    args = parse_args()
    tid_target = args.tid_target
    tid_source = args.tid_source
    nrun = args.nrun
    npilot = args.npilot
    nbatch = args.nbatch
    tuning_method = args.tuning_method

    if tuning_method == "SLA":
        tuning_problem_name = "Branin-tuning_method_"+str(tuning_method)+"-tid_target_"+str(tid_target)+"-npilot_"+str(npilot)+"-batch_num_"+str(nbatch)
    else:
        tuning_problem_name = "Branin-tuning_method_"+str(tuning_method)+"-tid_target_"+str(tid_target)+"-tid_source_"+str(tid_source)+"-npilot_"+str(npilot)+"-batch_num_"+str(nbatch)

    tuning_metadata = {
        "tuning_problem_name": tuning_problem_name,
        "sync_crowd_repo": "no",
        "no_load_check": "yes",
        "save_model": "no"
    }

    a_min, a_max = 0.5, 1.5
    a = Real(a_min, a_max, transform="normalize", name="a")
    b_min, b_max = 0.1, 0.15
    b = Real(b_min, b_max, transform="normalize", name="b")
    c_min, c_max = 1.0, 2.0
    c = Real(c_min, c_max, transform="normalize", name="c")
    r_min, r_max = 5.0, 7.0
    r = Real(r_min, r_max, transform="normalize", name="r")
    s_min, s_max = 8.0, 12.0
    s = Real(s_min, s_max, transform="normalize", name="s")
    t_min, t_max = 0.03, 0.05
    t = Real(t_min, t_max, transform="normalize", name="t")

    x1 = Real(-5.0, 10.0, transform="normalize", name="x1")
    x2 = Real(0.0, 15.0, transform="normalize", name="x2")

    y = Real(float('-Inf'), float('Inf'), name='y')

    constraints = {}

    input_space = Space([a,b,c,r,s,t])
    parameter_space = Space([x1, x2])
    output_space = Space([y])

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)

    with open("tasklist.json", "r") as f_in:
        target_task = json.load(f_in)["target_tasks"][tid_target]
        a_value = target_task["a"]
        b_value = target_task["b"]
        c_value = target_task["c"]
        r_value = target_task["r"]
        s_value = target_task["s"]
        t_value = target_task["t"]

    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=1, cores=2, hosts=None)
    data = Data(problem)
    options = Options()

    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_random_seed'] = nbatch
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = nbatch
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = nbatch

    if tuning_method == "SLA":
        options["TLA_method"] = None
        options.validate(computer=computer)
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        giventask = [a_value, b_value, c_value, r_value, s_value, t_value]
        (data, modeler, stats) = gt.SLA(NS=nrun, NS1=0, Tgiven=giventask)

        """ Print all input and parameter samples """
        print("stats: ", stats)
        print("    t: ", data.I)
        print("    Ps ", data.P)
        print("    Os ", data.O.tolist())
        print('    Popt ', data.P[np.argmin(data.O)], 'Oopt ', min(data.O)[0], 'nth ', np.argmin(data.O))

    else:
        if tuning_method == "TLA_Sum":
            options["TLA_method"] = "Sum"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Regression":
            options["TLA_method"] = "Regression"
            options.validate(computer=computer)
        elif tuning_method == "TLA_LCM_BF":
            options["TLA_method"] = "LCM_BF"
            options.validate(computer=computer)
        elif tuning_method == "TLA_LCM":
            options["TLA_method"] = "LCM"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Stacking":
            options["TLA_method"] = "Stacking"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Toggling":
            options["TLA_method"] = "Ensemble_Toggling"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Peeking":
            options["TLA_method"] = "Ensemble_Peeking"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Prob1":
            options["TLA_method"] = "Ensemble_Prob"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Prob2":
            options["TLA_method"] = "Ensemble_Prob"
            options["TLA_ensemble_exploration_rate"] = 0.5
            options.validate(computer=computer)
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        giventask = [[a_value, b_value, c_value, r_value, s_value, t_value]]
        (data, modeler, stats) = gt.TLA_I(NS=nrun, Tnew=giventask, source_function_evaluations=LoadSourceFunctionEvaluations(tid_source=tid_source))

        """ Print all input and parameter samples """
        print("stats: ", stats)
        for i in range(len(giventask)):
            """ Print all input and parameter samples """
            print("stats: ", stats)
            print("    t: ", data.I[i])
            print("    Ps ", data.P[i])
            print("    Os ", data.O[i].tolist())
            print('    Popt ', data.P[i][np.argmin(data.O[0])], 'Oopt ', min(data.O[i])[0], 'nth ', np.argmin(data.O[i]))

if __name__ == "__main__":
    main()
