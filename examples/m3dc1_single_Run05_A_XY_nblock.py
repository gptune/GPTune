
#! /usr/bin/env python3

"""
Example of invocation of this script:
python m3dc1_single.py -nodes 1 -cores 32 -ntask 20 -nrun 800 -machine cori

where:
    -nodes is the number of compute nodes
    -cores is the number of cores per node
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -machine is the name of the machine
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


################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
    
	COLPERM = point['COLPERM']
	ROWPERM = point['ROWPERM']
	nprows = 2**point['nprows']
	nproc = 2**point['nproc']*3
	# nproc = 12
	NSUP = point['NSUP']
	NREL = point['NREL']
	nblock     = int(nprocmax/nproc)
	npcols     = int(nproc/ nprows)
	params = ['ROWPERM', ROWPERM, 'COLPERM', COLPERM, 'nprows', nprows, 'npcols', npcols, 'NSUP', NSUP, 'NREL', NREL]

	# INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
	TUNER_NAME = os.environ['TUNER_NAME']

	nthreads   = 1


	""" pass some parameters through environment variables """	


	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=1\n'
	envstr+= 'NREL=%d\n' %(NREL)   
	envstr+= 'NSUP=%d\n' %(NSUP)   
	info.Set('env',envstr)


	fin = open("./options_bjacobi","rt")
	fout = open("./options_bjacobi_tune","wt")

	for line in fin:
		#read replace the string and write to output file
		if(line.find("-hard_pc_bjacobi_blocks")!=-1):
			fout.write("-hard_pc_bjacobi_blocks %s\n"%(nblock))
		elif(line.find("-pc_bjacobi_blocks")!=-1):
			fout.write("-pc_bjacobi_blocks %s\n"%(nblock))            
		else:
			fout.write(line)
	#close input and output files
	fin.close()
	fout.close()



	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "./m3dc1_3d_spawn", 'args', ['-ipetsc', '-options_file', 'options_bjacobi_tune', '-log_view', '-mat_superlu_dist_r', '%s'%(nprows), '-mat_superlu_dist_c', '%s'%(npcols), '-mat_superlu_dist_colperm', '%s'%(COLPERM), 'nblock', '%s'%(nblock)], 'nproc', nprocmax, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL))
	comm = MPI.COMM_SELF.Spawn("./m3dc1_3d_spawn", args=['-ipetsc', '-options_file', 'options_bjacobi_tune', '-log_view', '-mat_superlu_dist_r', '%s'%(nprows), '-mat_superlu_dist_c', '%s'%(npcols), '-mat_superlu_dist_colperm', '%s'%(COLPERM)], maxprocs=nprocmax,info=info)


	# retval=1.0


	""" gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """																	
	tmpdata = np.array([0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	
	retval = tmpdata[0]
	print(params, ' m3dc1 time: ', retval)

	return retval 
	
	
def main():

	global ROOTDIR
	global nodes
	global cores
	global target
	global nprocmax
	global nprocmin

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
	# JOBID = args.jobid


	os.environ['MACHINE_NAME']=machine
	os.environ['TUNER_NAME']='GPTune'
	TUNER_NAME = os.environ['TUNER_NAME']


	# nprocmax = nodes*cores-1
	nprocmax = 384
	nprocmin = nodes

	# Input parameters
	ROWPERM   = Categoricalnorm (['NOROWPERM', 'LargeDiag_MC64'], transform="onehot", name="ROWPERM")
	COLPERM   = Categoricalnorm (['MMD_AT_PLUS_A', 'METIS_AT_PLUS_A'], transform="onehot", name="COLPERM")
	nprows    = Integer     (0, 3, transform="normalize", name="nprows")
	nproc    = Integer     (0, 3, transform="normalize", name="nproc")
	NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
	NREL      = Integer     (10, 40, transform="normalize", name="NREL")	

	result   = Real        (float("-Inf") , float("Inf"), transform="normalize", name="r")



	examples = ["Run05_A_XY"]
	example    = Categoricalnorm (examples, transform="onehot", name="matrix")
	
	IS = Space([example])
	PS = Space([ROWPERM, COLPERM, nprows, nproc, NSUP, NREL])
	# PS = Space([ROWPERM, COLPERM, nprows, NSUP, NREL])
	OS = Space([result])
	cst1 = "NSUP >= NREL"
	cst2 = "nprows<=2"
	constraints = {"cst1" : cst1,"cst2" : cst2}
	models = {}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

	BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/M3DC1/m3dc1_newrepo_620_04_08_20/M3DC1/unstructured_openmpi_spawn_intel/_cori_knl-3d-opt-60")
	RUNDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/M3DC1/m3dc1_newrepo_620_04_08_20/M3DC1/Run05_A_XY")
	os.system("cp %s/*.smb ."%(RUNDIR))
	os.system("cp %s/C1input ."%(RUNDIR))
	os.system("cp %s/AnalyticModel ."%(RUNDIR))
	os.system("cp %s/options_bjacobi ."%(RUNDIR))
	os.system("cp %s/m3dc1_3d ./m3dc1_3d_spawn"%(BINDIR))





	# target='memory'
	target='time'


	problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
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
	options['model_class '] = 'Model_LCM'
	options['verbose'] = True

	options.validate(computer = computer)


	""" Intialize the tuner with existing data"""		
	data = Data(problem)
	gt = GPTune(problem, computer = computer, data = data, options = options)



	# """ Building MLA with NI random tasks """
	# NI = ntask
	# NS = nruns
	# (data, model,stats) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))
	# print("stats: ",stats)

	""" Building MLA with the given list of tasks """	
	giventask = [["Run05_A_XY"]]	
	NI = len(giventask)
	NS = nruns
	(data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = max(NS//2,1))
	# (data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = 10)
	print("stats: ",stats)


	""" Print all input and parameter samples """	
	for tid in range(NI):
		print("tid: %d"%(tid))
		print("    example:%s"%(data.I[tid][0]))
		print("    Ps ", data.P[tid])
		print("    Os ", data.O[tid])
		print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Yopt ', min(data.O[tid])[0])



def parse_args():

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
	return args


if __name__ == "__main__":
 
	main()
