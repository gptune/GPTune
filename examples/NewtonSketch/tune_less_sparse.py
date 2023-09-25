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
sys.path.insert(0, os.path.abspath(__file__ + "/newtonsketch/"))

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import *

import argparse
import numpy as np
import time

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores
from sklearn.kernel_approximation import RBFSampler
from solvers_lr import LogisticRegression
import generate_dataset

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-dataset', type=str, default='cifar-10', help='Dataset')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')

    args = parser.parse_args()

    return args

def objectives(point):
    dataset = point['dataset']
    sketch = point['sketch']
    n = point["n"]
    d = point["d"]
    m = int(d*point['sketch_size'])
    nnz = point['sparsity_parameter']*point["d"]/point["n"]
    lambd = point['lambd']
    error_threshold = point['error_threshold']
    niter = point['niter']

    print ("Dataset: ", dataset, "n: ", n, "d: ", d, "sketch: ", sketch, "lambda: ", lambd, "m: ", m, "nnz: ", nnz, "error_threshold: ", error_threshold, "niter: ", niter)

    times_spent = []
    for i in range(niter):
        _, losses_, times_ = lreg.ihs_tuning(sketch_size=m, sketch=sketch, nnz=nnz, error_threshold=error_threshold)

        print (losses_)
        print (times_)

        time_spent = times_[-1]
        times_spent.append(time_spent)
        loss_final = losses_[-1]

    return [times_spent]

def cst1(sketch_size, n, d):
    num_sketch_rows = int(d*sketch_size)
    return num_sketch_rows >= 1 and num_sketch_rows <= n

def cst2(sparsity_parameter, n, d):
    nnzs_per_row = int(sparsity_parameter*d/n*n)
    return nnzs_per_row >= 1 and nnzs_per_row <= n

def main():

    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    dataset = args.dataset
    nrun = args.nrun
    npilot = args.npilot
    TUNER_NAME = args.optimization

    tuning_metadata = {
        "tuning_problem_name": "less_sparse-"+dataset,
        "machine_configuration": {
            "machine_name": "Cori",
            "haswell": { "nodes": 1, "cores": 32 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "Cori": {
                "haswell": {
                    "nodes": 1,
                    "cores": 32
                }
            }
        },
        "loadable_software_configurations": {}
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    global A, b, lreg
    if dataset == 'cifar-10':
        A, b = generate_dataset.load_data('cifar-10', n=2**14, d=1000)
        lambd = 1e-4
        error_threshold = 1e-6
    elif dataset == 'synthetic_high_coherence_10000_2000':
        A, b = generate_dataset.load_data('synthetic_high_coherence', n=10000, d=2000, df=1)
        lambd = 1e-4
        error_threshold = 1e-6
    elif dataset == 'synthetic_high_coherence_20000_2000':
        A, b = generate_dataset.load_data('synthetic_high_coherence', n=20000, d=2000, df=1)
        lambd = 1e-4
        error_threshold = 1e-6
    elif dataset == 'synthetic_high_coherence_100000_2000':
        A, b = generate_dataset.load_data('synthetic_high_coherence', n=100000, d=2000, df=1)
        lambd = 1e-4
        error_threshold = 1e-6
    elif dataset == "epsilon_normalized_20Kn_spread":
        A, b = generate_dataset.load_data('epsilon_normalized_20Kn', option="spread")
        lambd = 1e-4
        error_threshold = 1e-6
    elif dataset == "epsilon_normalized_100Kn_spread":
        A, b = generate_dataset.load_data('epsilon_normalized_100Kn', option="spread")
        lambd = 1e-4
        error_threshold = 1e-6
    else:
        A, b = generate_dataset.load_data('synthetic_orthogonal')
        lambd = 1e-4
        error_threshold = 1e-6
    n, d = A.shape
    niter = 3
    lreg = LogisticRegression(A, b, lambd)
    x, losses = lreg.solve_exactly(n_iter=20, eps=1e-15)

    datasets = Categoricalnorm([dataset], transform="onehot", name="dataset")

    sketch = Categoricalnorm(["less_sparse"], transform="onehot", name="sketch")
    if "susy" in dataset:
        sketch_size = Real(1./d, 1000, transform="normalize", name="sketch_size")
    elif "synthetic_high_coherence" in dataset:
        sketch_size = Real(1./d, 0.1, transform="normalize", name="sketch_size")
    elif "epsilon" in dataset:
        sketch_size = Real(1./d, 2.0, transform="normalize", name="sketch_size")
    else:
        sketch_size = Real(1./d, n/d, transform="normalize", name="sketch_size")
    sparsity_parameter = Real(1./d, n/d, transform="normalize", name="sparsity_parameter")
    wall_clock_time = Real(float("-Inf"), float("Inf"), name="wall_clock_time")

    input_space = Space([datasets])
    parameter_space = Space([sketch, sketch_size, sparsity_parameter])
    output_space = Space([wall_clock_time])
    constraints = {"cst1": cst1, "cst2": cst2}

    constants={"n":n, "d":d, "lambd":lambd, "error_threshold":error_threshold, "niter":niter}

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None, constants=constants)
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
    options['model_class'] = 'Model_GPy_LCM'
    options['verbose'] = False
    options['sample_class'] = 'SampleOpenTURNS'
    options.validate(computer=computer)

    TUNER_NAME = os.environ['TUNER_NAME']

    giventask = [[dataset]]
    NI=len(giventask)
    NS=nrun

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        (data, modeler, stats) = gt.MLA(NS=NS, Tgiven=giventask, NI=NI, NS1=npilot)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%s " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()
