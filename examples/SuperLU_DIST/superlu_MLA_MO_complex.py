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
mpirun -n 1 python superlu_MLA_complex.py -nprocmin_pernode 1 -ntask 20 -nrun 800 

where:
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
"""
 
################################################################################

import sys
import os
import numpy as np
import argparse
import pickle
import copy

import mpi4py
from mpi4py import MPI
from array import array
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all



from autotune.problem import *
from autotune.space import *
from autotune.search import *
import pygmo as pg
import math

################################################################################


def objectives(point):                  # should always use this name for user-defined objective function
    
	matrix = point['matrix']
	COLPERM = point['COLPERM']
	LOOKAHEAD = point['LOOKAHEAD']
	nprows = point['nprows']
	npernode = 2**point['npernode']
	nproc = nodes*npernode
	nthreads = int(cores / npernode)
	NSUP = point['NSUP']
	NREL = point['NREL']
	npcols     = int(nproc / nprows)
	params = [matrix, 'COLPERM', COLPERM, 'LOOKAHEAD', LOOKAHEAD, 'nthreads', nthreads, 'npernode', npernode, 'nprows', nprows, 'npcols', npcols, 'NSUP', NSUP, 'NREL', NREL]
	RUNDIR = os.path.abspath(__file__ + "/../superlu_dist/build/EXAMPLE")
	INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
	TUNER_NAME = os.environ['TUNER_NAME']
	nproc     = int(nprows * npcols)

	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	envstr+= 'NREL=%d\n' %(NREL)   
	envstr+= 'NSUP=%d\n' %(NSUP)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works


	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "%s/pzdrive_spawn"%(RUNDIR), 'args', ['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL)  )
	comm = MPI.COMM_SELF.Spawn("%s/pzdrive_spawn"%(RUNDIR), args=['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator, also refer to the INPUTDIR/pzdrive_spawn.c to see how the return value are communicated """																	
	tmpdata = array('f', [0,0])
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.FLOAT],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()

	print(params, ' superlu factor time: ', tmpdata[0], ' solve time: ', tmpdata[1])
	# tmpdata1 = copy.deepcopy(tmpdata)
	# tmpdata1[0]=tmpdata[1]
	# tmpdata1[1]=tmpdata[0]	
	# return tmpdata1 
	return tmpdata 

	
def cst1(NSUP,NREL):
	return NSUP >= NREL
def cst2(npernode,nprows):
	return nodes * 2**npernode >= nprows		
def main():

	global ROOTDIR
	global nodes
	global cores
	global target
	global nprocmax

	# Parse command line arguments
	args   = parse_args()

	# Extract arguments
	ntask = args.ntask
	nprocmin_pernode = args.nprocmin_pernode
	optimization = args.optimization
	nrun = args.nrun
	TUNER_NAME = args.optimization
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))


	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME


	nprocmax = nodes*cores
	# matrices = ["big.rua", "g4.rua", "g20.rua"]
	matrices = ["nimrodMatrix-V.bin", "nimrodMatrix-B.bin", "cg20.cua"]
	# matrices = ["Si2.rb", "SiH4.rb", "SiNa.rb", "Na5.rb", "benzene.rb", "Si10H16.rb", "Si5H12.rb", "SiO.rb", "Ga3As3H12.rb", "GaAsH6.rb", "H2O.rb"]
	# Task parameters
	matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")
	# Input parameters
	COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
	LOOKAHEAD = Integer     (5, 20, transform="normalize", name="LOOKAHEAD")
	nprows    = Integer     (1, nprocmax, transform="normalize", name="nprows")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
	NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
	NREL      = Integer     (10, 40, transform="normalize", name="NREL")	
	factor   = Real        (float("-Inf") , float("Inf"), name="factor_time")
	solve    = Real        (float("-Inf") , float("Inf"), name="solve_time")
	IS = Space([matrix])
	PS = Space([COLPERM, LOOKAHEAD, npernode, nprows, NSUP, NREL])
	OS = Space([factor, solve])
	constraints = {"cst1" : cst1, "cst2" : cst2}
	models = {}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

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
	options['verbose'] = False
	options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso' 
	options['search_pop_size'] = 1000 # 1000
	options['search_gen'] = 10
	options['search_more_samples'] = 4
	options.validate(computer = computer)

	""" Intialize the tuner with existing data"""		
	data = Data(problem)
	gt = GPTune(problem, computer = computer, data = data, options = options, driverabspath=os.path.abspath(__file__))

	if(TUNER_NAME=='GPTune'):
		#""" Building MLA with NI random tasks """
		#NI = ntask
		#NS = nrun
		#(data, model,stats) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))
		#print("stats: ",stats)

		""" Building MLA with the given list of tasks """	
		#giventask = [["big.rua"], ["g4.rua"], ["g20.rua"]]	
		giventask = [["cg20.cua"]]	
		# giventask = [["nimrodMatrix-V.bin"]]	
		NI = len(giventask)
		NS = nrun
		(data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = max(NS//2,1))
		print("stats: ",stats)


		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid])
			ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data.O[tid])
			front = ndf[0]
			# print('front id: ',front)
			fopts = data.O[tid][front]
			xopts = [data.P[tid][i] for i in front]
			print('    Popts ', xopts)		
			print('    Oopts ', fopts)		
			
  
def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')

	args   = parser.parse_args()
	return args
	
if __name__ == "__main__":
	main()
