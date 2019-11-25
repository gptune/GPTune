#! /usr/bin/env python3

"""
Example of invocation of this script:

python superlu.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -modes is the number of compute nodes
    -cores is the number of cores per node
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the total number of superlu call.  The number of calls per task is thus nrun / ntask
    -machine is the name of the machine
    -jobid is optional. You can always set it to 0.
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))

from computer import Computer
from options import Options
from data import Data
from gptune import GPTune

from autotune.problem import *
from autotune.space import *
from autotune.search import *


# sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))
# from pdqrdriver import pdqrdriver

################################################################################


def objective(point):                  # should always use this name for user-defined objective function
    
	matrix = point['matrix']
	COLPERM = point['COLPERM']
	LOOKAHEAD = point['LOOKAHEAD']
	NTH = point['NTH']
	nprows = point['nprows']
	npcols = point['npcols']
	NSUP = point['NSUP']
	NREL = point['NREL']

  
	RUNDIR = os.path.abspath("/global/homes/l/liuyangz/Cori/my_research/github/superlu_dist_master_gptune_11_22_2019/build/EXAMPLE")

	outputfilename = os.path.abspath("/global/homes/l/liuyangz/Cori/my_research/github/superlu_dist_master_gptune_11_22_2019/exp/{tunername}/superlu_{matrix}_{COLPERM}_{LOOKAHEAD}_{NTH}_{nprows}_{npcols}_{NSUP}_{NREL}.out")
    

    
	info = MPI.Info.Create()
	info.Set('env', 'OMP_NUM_THREADS=%d\n' %(NTH))
	info.Set('env', 'NTH=%d\n' %(NTH))
	info.Set('env', 'NSUP=%d\n' %(NSUP))
	info.Set('env', 'NREL=%d\n' %(NREL))
	info.Set('env', 'OMP_PLACES=threads\n')
	info.Set('env', 'OMP_PROC_BIND=OMP_PROC_BIND\n')
    
    # info.Set("add-hostfile", "myhostfile.txt")
    # info.Set("host", "myhostfile.txt")
    
    # print('exec', "%s/pdqrdriver"%(BINDIR), 'args', "%s/"%(RUNDIR), 'nproc', nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
	comm = MPI.COMM_SELF.Spawn("%s/pddrive"%(RUNDIR), args="-c {npcols} -r {nprows} -l {LOOKAHEAD} -p {COLPERM} ../EXAMPLE/{matrix} > {outputfilename}"%(RUNDIR), maxprocs=nproc,info=info)
	comm.Disconnect()	
	
	
	elapsedtime = float(1e3)
	f = open(outputfilename, 'r')
	for line in f.readlines():
		s = line.split()
		if (len(s) >= 3 and s[0] == "FACTOR" and s[1] == "time"):
			elapsedtime = float(s[2])
			break

	print(matrix, COLPERM, LOOKAHEAD, NTH, nprows, npcols, NSUP, NREL, elapsedtime)

	print(params, ' superlu time: ', elapsedtime)

	return elapsedtime 

	
	
def main_interactive():

    global ROOTDIR
    global nodes
    global cores

    # Parse command line arguments

    parser = argparse.ArgumentParser()

    # Problem related arguments
    # parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    # parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)

    args   = parser.parse_args()

    # Extract arguments

    # mmax = args.mmax
    # nmax = args.nmax
    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    optimization = args.optimization
    nruns = args.nruns
    truns = args.truns
    JOBID = args.jobid
    
    
    os.environ['MACHINE_NAME']=machine
    os.environ['TUNER_NAME']='GPTune'
    TUNER_NAME = os.environ['TUNER_NAME']
    # print(os.environ)
    os.system("mkdir -p /global/homes/l/liuyangz/Cori/my_research/github/superlu_dist_master_gptune_11_22_2019/exp; mkdir -p /global/homes/l/liuyangz/Cori/my_research/github/superlu_dist_master_gptune_11_22_2019/exp/%s;"%(TUNER_NAME))

# YL: for the spaces, the following datatypes are supported: Note: Categorical's onehot transform has not been tested
# Real(lower, upper, "uniform", "normalize", name="yourname")
# Integer(lower, upper, "normalize", name="yourname")
# Categorical(categories, transform="onehot", name="yourname")  	
	
	
    matrices = ["big.rua", "g4.rua", "g20.rua"]
    # matrices = ["Si2.rb", "SiH4.rb", "SiNa.rb", "Na5.rb", "benzene.rb", "Si10H16.rb", "Si5H12.rb", "SiO.rb", "Ga3As3H12.rb", "GaAsH6.rb", "H2O.rb"]

    # Task parameters
    matrix    = Categorical (matrices, name="matrix")

    # Input parameters
    COLPERM   = Categorical ([2, 4], name="COLPERM")
    LOOKAHEAD = Integer     (5, 20, "normalize", name="LOOKAHEAD")
    NTH       = Integer     (1, cores, "normalize", name="NTH")
    nprows    = Integer     (1, nodes*cores, "normalize", name="nprows")
    npcols    = Integer     (1, nodes*cores, "normalize", name="npcols")
    NSUP      = Integer     (30, 300, "normalize", name="NSUP")
    NREL      = Integer     (10, 40, "normalize", name="NREL")	
    runtime   = Real        (float("-Inf") , float("Inf"), "normalize", name="r")
	
    IS = Space([matrix])
    PS = Space([COLPERM, LOOKAHEAD, NTH, nprows, npcols, NSUP, NREL])
    OS = Space([runtime])


    cst1 = "NTH * nprows * npcols == {nodes} * {cores}"
    cst2 = "NSUP >= NREL"

    constraints = {"cst1" : cst1, "cst2" : cst2}
    models = {}

    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objective, constraints, None)
    print("inimaa",nodes)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)  

    options = Options()
    options['model_processes'] = 1
    options['model_threads'] = 1
    options['model_restarts'] = 1
    options['search_multitask_processes'] = 1
    options['model_restart_processes'] = 1
    options['distributed_memory_parallelism'] = True
    options['shared_memory_parallelism'] = False
    options['mpi_comm'] = None
    options['model_class '] = 'Model_LCM'
    data = Data(problem)
    gt = GPTune(problem, computer = computer, data = data, options = options)



    NI = ntask
    NS = nruns

    (data, model) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))

    # for tid in range(NI):
        # print("tid: %d"%(tid))
        # print("    m:%d n:%d"%(data.T[tid][0], data.T[tid][1]))
        # print("    Xs ", data.X[tid])
        # print("    Ys ", data.Y[tid])
        # print('    Xopt ', data.X[tid][np.argmin(data.Y[tid])], 'Yopt ', min(data.Y[tid])[0])
		
		
def parse_args():

    # Parse command line arguments

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')

    args   = parser.parse_args()

    # Extract arguments

    return (args.mmax, args.nmax, args.ntask, args.nodes, args.cores, args.machine, args.optimization, args.nruns, args.truns, args.jobid, args.stepid, args.phase)


if __name__ == "__main__":
 
   main_interactive()