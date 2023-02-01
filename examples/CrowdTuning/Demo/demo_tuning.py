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
sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))

from gptune import *

import argparse
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=100, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')
    parser.add_argument('-nbatch', type=int, default=0, help='Input task t value')
    parser.add_argument('-tuning_method', type=str, default='SLA', help='Tuning method')

    args = parser.parse_args()

    return args

def LoadFunctionEvaluations():
    api_key = os.getenv("CROWDTUNING_API_KEY")

    import crowdtune
    problem_space = {
        "input_space": [
            {"name":"t", "type":"real", "transformer":"normalize", "lower_bound":1.0, "upper_bound":1.001}
        ],
        "parameter_space": [
            {"name":"x", "type":"real", "transformer":"normalize", "lower_bound":0.0, "upper_bound":1.0}
        ],
        "output_space": [
            {"name":"y", "type":"real", "transformer":"identity", "lower_bound":float('-Inf'), "upper_bound":float('Inf')}
        ]
    }

    function_evaluations = crowdtune.QueryFunctionEvaluations(api_key = api_key,
        tuning_problem_name = "GPTune-Demo",
        problem_space = problem_space)

    return [function_evaluations]

def LoadModels():
    api_key = os.getenv("CROWDTUNING_API_KEY")

    import crowdtune
    problem_space = {
        "input_space": [
            {"name":"t", "type":"real", "transformer":"normalize", "lower_bound":1.0, "upper_bound":1.001}
        ],
        "parameter_space": [
            {"name":"x", "type":"real", "transformer":"normalize", "lower_bound":0.0, "upper_bound":1.0}
        ],
        "output_space": [
            {"name":"y", "type":"real", "transformer":"identity", "lower_bound":float('-Inf'), "upper_bound":float('Inf')}
        ]
    }

    surrogate_model = crowdtune.QuerySurrogateModel(
        api_key = api_key,
        tuning_problem_name = "GPTune-Demo",
        problem_space = problem_space,
        input_task = [1.0])

    return [surrogate_model]

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

    return [f]

def main():

    global tvalue

    args = parse_args()
    tvalue = args.tvalue
    nrun = args.nrun
    npilot = args.npilot
    nbatch = args.nbatch
    tuning_method = args.tuning_method

    tuning_metadata = {
        "tuning_problem_name": "GPTune-Demo-"+str(tuning_method)+"-"+str(tvalue)+"-"+str(nbatch)+"-npilot"+str(npilot),
        "sync_crowd_repo": "no",
        "no_load_check": "yes",
        "machine_configuration": {
            "machine_name": "Cori",
            "haswell": { "nodes": 1, "cores": 32 }
        },
        "software_configuration": {}
    }

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=1, cores=32, hosts=None)
    data = Data(problem)

    options = Options()
    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_random_seed'] = nbatch
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = nbatch
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = nbatch

    if tuning_method == "TLA_Sum":
        options['TLA_method'] = "Sum"
    elif tuning_method == "TLA_Regression":
        options['TLA_method'] = "Regression"
    elif tuning_method == "TLA_LCM_BF":
        options['TLA_method'] = "LCM_BF"
    elif tuning_method == "TLA_LCM_GPY":
        options['TLA_method'] = "LCM"

    if tuning_method == "TLA_Regression":
        options['regression_log_name'] = "GPTune-Demo-"+str(tuning_method)+"-"+str(tvalue)+"-"+str(nbatch)+"-npilot"+str(npilot) + "-models-weights.log"

    options['sample_class'] = 'SampleOpenTURNS'

    options.validate(computer=computer)

    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

    if tuning_method == "SLA":
        giventask = [[round(tvalue,2)]]
        (data, model, stats) = gt.MLA(NS=nrun, NI=len(giventask), Tgiven=giventask, NS1=npilot)
    else:
        giventask = [[round(tvalue,2)]]
        (data, modeler, stats) = gt.TLA_I(NS=nrun, Tnew=giventask, models_transfer=LoadModels(), source_function_evaluations=LoadFunctionEvaluations())

    """ Print all input and parameter samples """
    print("stats: ", stats)
    for tid in range(len(giventask)):
        print("tid: %d" % (tid))
        print("    t:%f " % (data.I[tid][0]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()
