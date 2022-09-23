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

python ./demo.py -nrun 20 npilot 0 -tvalue 1.0

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

    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=0, help='Number of initial runs per task')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')

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

    return [f]

def main():

    global tvalue

    args = parse_args()
    tvalue = args.tvalue
    nrun = args.nrun
    npilot = args.npilot

    tuning_metadata = {
        "tuning_problem_name": "Demo-Tutorial",
        "crowdtuning_api_key": os.getenv("CROWDTUNING_API_KEY"),
        "sync_crowd_repo": "yes",
        "load_func_eval": "no",
        "machine_configuration": {
            "machine_name": "mymachine",
            "haswell": { "nodes": 1, "cores": 2 }
        },
        "software_configuration": {}
    }

    import openturns as ot
    ot.RandomGenerator.SetSeed(0)
    print(args)

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}

    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=1, cores=2, hosts=None)
    data = Data(problem)

    options = Options()
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['sample_class'] = 'SampleOpenTURNS'
    options.validate(computer=computer)

    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

    giventask = [[round(tvalue,2)]]
    (data, model, stats) = gt.MLA(NS=nrun, NI=len(giventask), Tgiven=giventask, NS1=npilot)

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
