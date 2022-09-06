#! /usr/bin/env python3

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

"""
Example of invocation of this script:
mpirun -n 8 python superlu_TLA_base.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -obj time

where:
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
"""

import sys
import os
import numpy as np
import argparse

import mpi4py
from mpi4py import MPI
from array import array
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))

from gptune import *
from autotune.problem import *
from autotune.space import *
from autotune.search import *

from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math
import time

def objectives(point): # should always use this name for user-defined objective function

    print ("POINT: ", point)

    nodes = point['nodes']
    cores = point['cores']

    matrix = point['matrix']
    COLPERM = point['COLPERM']
    LOOKAHEAD = point['LOOKAHEAD']
    nprows = point['nprows']

    npernode = point['npernode']
    nproc = nodes*npernode
    nthreads = int(cores / npernode)

    NSUP = point['NSUP']
    NREL = point['NREL']
    npcols = int(nproc / nprows)
    params = [matrix, 'COLPERM', COLPERM, 'LOOKAHEAD', LOOKAHEAD, 'nthreads', nthreads, 'npernode', npernode, 'nprows', nprows, 'npcols', npcols, 'NSUP', NSUP, 'NREL', NREL]
    RUNDIR = os.path.abspath(__file__ + "/../superlu_dist/build/EXAMPLE")
    INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
    nproc = int(nprows * npcols)

    """ pass some parameters through environment variables """
    obj_arr = []
    runtime_arr = []
    memory_arr = []
    for i in range(5):
        info = MPI.Info.Create()
        envstr = 'OMP_NUM_THREADS=%d\n' %(nthreads)
        envstr += 'NREL=%d\n' %(NREL)
        envstr += 'NSUP=%d\n' %(NSUP)
        info.Set('env',envstr)
        info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works

        """ use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
        print('exec', "%s/pddrive_spawn"%(RUNDIR), 'args', ['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL))
        comm = MPI.COMM_SELF.Spawn("%s/pddrive_spawn"%(RUNDIR), args=['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], maxprocs=nproc,info=info)

        """ gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """
        tmpdata = array('f', [0,0])
        comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.FLOAT],op=MPI.MAX,root=mpi4py.MPI.ROOT)
        comm.Disconnect()
        time.sleep(5.0)

        runtime = tmpdata[0]
        memory = tmpdata[1]
        obj_arr.append(runtime)
        runtime_arr.append(runtime)
        memory_arr.append(memory)
        print(params, ' superlu time: ', runtime)
        print(params, ' superlu memory: ', memory)

    obj_arr.remove(max(obj_arr))
    obj_arr.remove(min(obj_arr))
    result = np.average(obj_arr)

    return [result], { "runtime": runtime_arr, "memory": memory_arr }

def cst1(NSUP,NREL):
    return NSUP >= NREL

def cst2(npernode,nprows,nodes):
    return nodes * npernode >= nprows

def main():
    # Parse command line arguments
    args   = parse_args()

    # Extract arguments
    ntask = args.ntask
    nprocmin_pernode = args.nprocmin_pernode
    optimization = args.optimization
    matname = args.matname
    nbatch = args.nbatch
    nrun = args.nrun
    npilot = args.npilot

    tuning_metadata = {
        "tuning_problem_name": "SuperLU_DIST-pddrive_spawn-"+str(matname)+"-"+str(nbatch),
        "use_crowd_repo": "no",
        "no_load_check": "yes",
        "machine_configuration": {
            "machine_name": "Cori",
            "slurm": "yes"
        },
        "software_configuration": {
            "openmpi": {
                "version_split": [4,0,1]
            },
            "parmetis": {
                "version_split": [4,0,3]
            },
            "superlu_dist": {
                "version_split": [6,4,0]
            }
        }
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine

    nprocmax = nodes*cores
    matrices = ["Si2.mtx", "SiH4.mtx", "SiNa.mtx", "benzene.mtx", "Na5.mtx", "Si5H12.mtx", "Si10H16.mtx", "SiO.mtx", "H2O.mtx", "GaAsH6.mtx", "Ga3As3H12.mtx"]
    # Task parameters
    matrix = Categoricalnorm (matrices, transform="onehot", name="matrix")
    # Input parameters
    COLPERM = Categoricalnorm (['1','2','3','4','5'], transform="onehot", name="COLPERM")
    LOOKAHEAD = Integer(5, 20, transform="normalize", name="LOOKAHEAD")
    nprows = Integer(1, int(math.sqrt(nprocmax)), transform="normalize", name="nprows")
    NSUP = Integer(30, 300, transform="normalize", name="NSUP")
    NREL = Integer(10, 40, transform="normalize", name="NREL")
    result = Real(float("-Inf"), float("Inf"), name="runtime")
    #result = Real(float("-Inf"), float("Inf"), name="memory")
    IS = Space([matrix])
    PS = Space([COLPERM, LOOKAHEAD, nprows, NSUP, NREL])
    OS = Space([result])
    constraints = {"cst1" : cst1, "cst2" : cst2}
    models = {}
    constants= {"nodes":nodes, "cores":cores, "npernode":cores}

    """ Print all input and parameter samples """
    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['model_class'] = 'Model_GPy_LCM'
    options['sample_class'] = 'SampleOpenTURNS'
    options['verbose'] = False
    options.validate(computer = computer)

    # """ Building MLA with the given list of tasks """
    giventask = [[matname]]
    data = Data(problem)

    #gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

    NI = len(giventask)
    NS = nrun
    NS1 = npilot
    (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=npilot)
    print("stats: ", stats)

    """ Print all input and parameter samples """
    for tid in range(NI):
        print("tid: %d"%(tid))
        print("    matrix:%s"%(data.I[tid][0]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-matname', type=str, default='Si2', help='Input matrix name')
    parser.add_argument('-nbatch', type=int, default=0, help='Batch run number')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=10, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=1, help='Number of runs per task')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
