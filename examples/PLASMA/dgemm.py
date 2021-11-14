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
import subprocess
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import *
import argparse
import numpy as np
import time

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=10, help='Number of initial runs per task')

    args = parser.parse_args()

    return args

def objectives(point):

    m = point["m"]
    n = point["n"]
    k = point["k"]
    nb = point["nb"]
    nthreads = point["nthreads"]
    pada = point["pada"]
    padb = point["padb"]
    padc = point["padc"]
    niter = point['niter']
    bind = point['bind']

    command = f"OMP_NUM_THREADS={nthreads} OMP_PROC_BIND={bind} plasmatest dgemm --iter={niter} --dim={m}x{n}x{k} --nb={nb} --pada={pada} --padb={padb} --padc={padc}"
    print (command)

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()
    print (output)
    print (errors)

    runtime = [float(output.split()[17 + 19 * i]) for i in range(niter)]
    #gflops = [float(output.split()[14 + 10 * i]) for i in range(niter)]

    return [runtime]

def cst1(nb, bunit):
    return nb%bunit == 0

def main():

    args = parse_args()
    nrun = args.nrun
    npilot = args.npilot

    tuning_metadata = {
        "tuning_problem_name": "DGEMM",
        "tuning_problem_category": "PLASMA",
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "Cori",
            "haswell": { "nodes": 1, "cores": 32 }
        },
        "spack": ["plasma"]
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    giventask = [[4096,4096,4096]]

    m = Integer(128, 4096, transform="normalize", name="m")
    n = Integer(128, 4096, transform="normalize", name="n")
    k = Integer(128, 4096, transform="normalize", name="k")
    nb = Integer(1, 1024, transform="normalize", name="nb")
    nthreads = Integer(1, 64, transform="normalize", name="nthreads")
    runtime = Real(float("-Inf"), float("Inf"), name="runtime")
    #gflops = Real(float("-Inf"), float("Inf"), name="gflops")

    input_space = Space([m, n, k])
    parameter_space = Space([nb, nthreads])
    output_space = Space([runtime])
    constraints = {"cst1": cst1}
    constants={"pada":0, "padb":0, "padc":0, "bunit":2, "niter":5, "bind":"true"}

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, constants=constants)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['model_class'] = 'Model_GPy_LCM'
    options['sample_class'] = 'SampleOpenTURNS'
    options.validate(computer=computer)

    NI=len(giventask)
    NS=nrun

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=npilot)
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
