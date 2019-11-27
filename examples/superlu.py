#! /usr/bin/env python3

"""
Example of invocation of this script:

python superlu.py -nodes 1 -cores 32 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -nodes is the number of compute nodes
    -cores is the number of cores per node
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -machine is the name of the machine
    -jobid is optional. You can always set it to 0.
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

import mpi4py
from mpi4py import MPI
from array import array

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))

from computer import Computer
from options import Options
from data import Data
from data import Categoricalnorm
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
	# NTH = point['NTH']
	nprows = point['nprows']
	nproc = point['nproc']
	# npcols = point['npcols']
	NSUP = point['NSUP']
	NREL = point['NREL']

	NTH   = int((nodes * cores-2) / nproc) # YL: there are at least 2 cores working on other stuff
	npcols     = int(nproc / nprows)


	params = [matrix, 'COLPERM', COLPERM, 'LOOKAHEAD', LOOKAHEAD, 'NTH', NTH, 'nprows', nprows, 'npcols', npcols, 'NSUP', NSUP, 'NREL', NREL]
  
	RUNDIR = os.path.abspath(__file__ + "/../superlu_dist/build/EXAMPLE")
	INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
	TUNER_NAME = os.environ['TUNER_NAME']
    
	# outputfilename = os.path.abspath("/global/homes/l/liuyangz/Cori/my_research/github/superlu_dist_master_gptune_11_22_2019/exp/%s/superlu_%s_%s_%s_%s_%s_%s_%s_%s.out"%(TUNER_NAME,matrix,COLPERM,LOOKAHEAD,NTH,nprows,npcols,NSUP,NREL))
    
	nproc     = int(nprows * npcols)
    
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(NTH)   
	envstr+= 'NREL=%d\n' %(NREL)   
	envstr+= 'NSUP=%d\n' %(NSUP)   
	info.Set('env',envstr)

	#info.Set('env', 'NTH=%d' %(NTH))
	#info.Set('env', 'NSUP=%d' %(NSUP))
	#info.Set('env', 'NREL=%d' %(NREL))
	# info.Set('env', 'OMP_PLACES=threads\n')
	# info.Set('env', 'OMP_PROC_BIND=OMP_PROC_BIND\n')
    
    # info.Set("add-hostfile", "myhostfile.txt")
    # info.Set("host", "myhostfile.txt")
    
	print('exec', "%s/pddrive_spawn"%(RUNDIR), 'args', ['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(NTH), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL)  )#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
	# comm = MPI.COMM_SELF.Spawn("%s/pddrive_spawn"%(RUNDIR), args="-c %s -r %s -l %s -p %s %s/%s"%(npcols,nprows,LOOKAHEAD,COLPERM,INPUTDIR,matrix), maxprocs=nproc,info=info)
	comm = MPI.COMM_SELF.Spawn("%s/pddrive_spawn"%(RUNDIR), args=['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], maxprocs=nproc,info=info)
	# (tmpdata) = comm.recv(source=0)	
	tmpdata = array('f', [0,0])
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.FLOAT],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	# print(tmpdata,'got it')
	comm.Disconnect()	


	retval = tmpdata[0]
	print(params, ' superlu time: ', retval)
 
	# retval = tmpdata[1]
	# print(params, ' superlu memory: ', retval)



	# retval = float(1e3)
	# f = open(outputfilename, 'r')
	# for line in f.readlines():
	# 	s = line.split()
	# 	if (len(s) >= 3 and s[0] == "FACTOR" and s[1] == "time"):
	# 		retval = float(s[2])
	# 		break

	# print(matrix, COLPERM, LOOKAHEAD, NTH, nprows, npcols, NSUP, NREL, elapsedtime)
	
	

	return retval 

	
	
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
    matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")

    # Input parameters
    COLPERM   = Categoricalnorm ([2, 4], transform="onehot", name="COLPERM")
    LOOKAHEAD = Integer     (5, 20, transform="normalize", name="LOOKAHEAD")
    nprows    = Integer     (1, nodes*cores, transform="normalize", name="nprows")
    nproc     = Integer     (nodes, nodes*cores, transform="normalize", name="nproc")
    NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
    NREL      = Integer     (10, 40, transform="normalize", name="NREL")	
    runtime   = Real        (float("-Inf") , float("Inf"), transform="normalize", name="r")
	
    IS = Space([matrix])
    PS = Space([COLPERM, LOOKAHEAD, nproc, nprows, NSUP, NREL])
    OS = Space([runtime])


    cst1 = "%d * %d"%(nodes, cores) + ">= nproc+2"  # YL: there are at least 2 cores working on other stuff
    cst2 = "NSUP >= NREL"
    cst3 = "nproc >= nprows" # intrinsically implies "p <= nproc"



    constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3}
    models = {}

    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objective, constraints, None)
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

    for tid in range(NI):
        print("tid: %d"%(tid))
        print("    matrix:%s"%(data.T[tid][0]))
        print("    Xs ", data.X[tid])
        print("    Ys ", data.Y[tid])
        print('    Xopt ', data.X[tid][np.argmin(data.Y[tid])], 'Yopt ', min(data.Y[tid])[0])
		
		
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