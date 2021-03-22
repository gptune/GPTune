#! /usr/bin/env python3

"""
Example of invocation of this script:

python scalapack_MLA_RCI.py -nodes 1 -cores 32 -nprocmin_pernode 1 -nruns 10 -machine cori -jobid 0 -bunit 8

where:
    -nodes is the number of compute nodes
    -cores is the number of cores per node
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -nruns is the number of calls per task 
    -machine is the name of the machine
    -jobid is optional. You can always set it to 0.
    -tla is whether TLA is used after MLA
"""

################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from options import Options
from computer import Computer
import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import time
import math


################################################################################

''' The objective function required by GPTune. '''
# should always use this name for user-defined objective function
def objectives(point):                          
	print('objective is not needed when options["RCI_mode"]=True')

def cst1(mb,p,m):
    return mb*bunit * p <= m
def cst2(nb,npernode,n,p):
    return nb * bunit * nodes * 2**npernode <= n * p
def cst3(npernode,p):
    return nodes * 2**npernode >= p

def main():

    global ROOTDIR
    global nodes
    global cores
    global bunit
    global JOBID
    global nprocmax
    global nprocmin

    # Parse command line arguments
    args = parse_args()

    nodes = args.nodes
    cores = args.cores
    bunit = args.bunit
    nprocmin_pernode = args.nprocmin_pernode
    machine = args.machine
    nruns = args.nruns
    tla = args.tla
    JOBID = args.jobid
    TUNER_NAME = args.optimization

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    os.system("mkdir -p scalapack-driver/bin/%s; cp ../../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))


    nprocmax = nodes*cores

    mmin=128
    nmin=128
    mmax=2000
    nmax=2000

    m = Integer(mmin, mmax, transform="normalize", name="m")
    n = Integer(nmin, nmax, transform="normalize", name="n")
    mb = Integer(1, 16, transform="normalize", name="mb")
    nb = Integer(1, 16, transform="normalize", name="nb")
    npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
    p = Integer(1, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, npernode, p])
    OS = Space([r])
    
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    # print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class'] = 'Model_LCM'
    options['verbose'] = False
    options['RCI_mode'] = True
    options.validate(computer=computer)

    # giventask = [[randint(mmin,mmax),randint(nmin,nmax)] for i in range(ntask)]
    # # giventask = [[2000, 2000]]
    # giventask = [[177, 1303],[367, 381],[1990, 1850],[1123, 1046],[200, 143],[788, 1133],[286, 1673],[1430, 512],[1419, 1320],[622, 263] ]
    giventask = [[177, 1303],[367, 381]]
    ntask=len(giventask)
    

    data = Data(problem)
    if(TUNER_NAME=='GPTune'):

        gt = GPTune(problem, computer=computer, data=data, options=options)

        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nruns
        (data, model, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=max(NS//2, 1))
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
        
        if(tla==1):
            """ Call TLA for 2 new tasks using the constructed LCM model"""
            newtask = [[400, 500], [800, 600]]
            (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-bunit', type=int, default=8, help='mb and nb are integer multiples of bunit')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments    
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-tla', type=int, default=0, help='Whether perform TLA after MLA when optimization is GPTune')    
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')



    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
