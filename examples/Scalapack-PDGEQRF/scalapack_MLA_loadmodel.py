#! /usr/bin/env python3

"""
Example of invocation of this script:

python scalapack.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -nodes is the number of compute nodes
    -cores is the number of cores per node
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -machine is the name of the machine
    -jobid is optional. You can always set it to 0.
"""

################################################################################

from pdqrdriver import pdqrdriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from options import Options
from computer import Computer
from historydb import HistoryDB
import sys
import os
import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import time
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))


sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))


################################################################################

''' The objective function required by GPTune. '''


def objectives(point):
    m = point['m']
    n = point['n']
    mb = point['mb']*8
    nb = point['nb']*8
    nproc = point['nproc']
    p = point['p']

    # this becomes useful when the parameters returned by TLA1 do not respect the constraints
    if(nproc == 0 or p == 0 or nproc < p):
        print('Warning: wrong parameters for objective function!!!')
        return 1e12
    npernode =  math.ceil(float(nproc)/nodes)
    nthreads = int(cores / npernode)
    q = int(nproc / p)
    params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]


    elapsedtime = pdqrdriver(params, niter=3, JOBID=JOBID)
    print(params, ' scalapack time: ', elapsedtime)

    return elapsedtime


def main():

    global ROOTDIR
    global nodes
    global cores
    global JOBID
    global nprocmax
    global nprocmin

    # Parse command line arguments
    args = parse_args()

    mmax = 32000
    nmax = 32000
    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    nprocmin_pernode = args.nprocmin_pernode
    machine = args.machine
    nrun = args.nrun
    truns = args.truns
    JOBID = args.jobid
    TUNER_NAME = 'GPTune'
    update = args.update

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores-1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
    nprocmin = min(nodes*nprocmin_pernode,nprocmax-1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space

    mmin=128
    nmin=128

    m = Integer(mmin, mmax, transform="normalize", name="m")
    n = Integer(nmin, nmax, transform="normalize", name="n")
    mb = Integer(1, 16, transform="normalize", name="mb")
    nb = Integer(1, 16, transform="normalize", name="nb")
    nproc = Integer(nprocmin, nprocmax, transform="normalize", name="nproc")
    p = Integer(0, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, nproc, p])
    OS = Space([r])
    cst1 = "mb*8 * p <= m"
    cst2 = "nb*8 * nproc <= n * p"
    cst3 = "nproc >= p"
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    # options['model_processes'] = 16
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class '] = 'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)

    data = Data(problem)

    # setting to invoke history database
    history_db = HistoryDB()
    history_db.history_db = 1
    history_db.application_name = 'scalapack-pdqrdriver'

    history_db.machine_deps = {
                "machine":machine,
                "nodes":nodes,
                "cores":cores
            }

    history_db.compile_deps = {
                "openmpi":{
                    "version":"4.0.0",
                    "version_split":[4,0,0],
                    "tags":"lib,mpi,openmpi"
                },
                "scalapack":{
                    "version":"2.1.0",
                    "version_split":[2,1,0],
                    "tags":"lib,scalapack"
                }
            }
    history_db.runtime_deps = {}

    # setting options for loading previous data
    # for now, task parameter has to be the same.
    history_db.load_deps = {
                "machine_deps": {
                    "machine":[machine],
                    "nodes":[nodes],
                    "cores":[cores]
                },
                "software_deps": {
                    "compile_deps": {
                        "mpi":[
                            {
                                "name":"openmpi",
                                "version_from":[4,0,0],
                                "version_to":[5,0,0]
                            }
                        ],
                        "scalapack":[
                            {
                                "name":"scalapack",
                                "version":[2,1,0]
                            }
                        ]
                    },
                    "runtime_deps": {}
                }
            }

    giventask = [[1024,1024],[2048,2048],[4096,4096],[8192,8192],[16384,16384]]

    gt = GPTune(problem, computer=computer, data=data, options=options, history_db=history_db)

    """ Building MLA with NI random tasks """
    NI = ntask
    NS = nrun
    #(data, model, stats) = gt.MLA_HistoryDB(NS=NS, Igiven=giventask, NI=NI, NS1=max(NS//2, 1))
    (data, model, stats) = gt.MLA_LoadModel(NS=NS, Igiven=giventask, update=update)
    print("stats: ", stats)

    """ Print all input and parameter samples """
    for tid in range(NI):
        print("tid: %d" % (tid))
        print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    with open("modeling_stat_scalapack.csv", "w") as f_out:
        for i in range(len(stats["modeling_time"])):
            f_out.write(str(i) + "," + str(stats["modeling_time"][i]) + "," + str(stats["modeling_iteration"][i]) + "\n")

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')
    parser.add_argument('-update', type=int, default=0, help='model update after loading model')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
