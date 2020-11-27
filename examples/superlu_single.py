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
python superlu.py -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 20 -nrun 800 -machine cori

where:
    -nodes is the number of compute nodes
    -cores is the number of cores per node
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
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
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))

from computer import Computer
from options import Options
from data import Data
from data import Categoricalnorm
from gptune import GPTune

from autotune.problem import *
from autotune.space import *
from autotune.search import *

from callopentuner import OpenTuner
from callhpbandster import HpBandSter
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
	print('exec', "%s/pddrive_spawn"%(RUNDIR), 'args', ['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL)  )
	comm = MPI.COMM_SELF.Spawn("%s/pddrive_spawn"%(RUNDIR), args=['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """																	
	tmpdata = array('f', [0,0])
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.FLOAT],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	

	if(target=='time'):	
		retval = tmpdata[0]
		print(params, ' superlu time: ', retval)

	if(target=='memory'):	
		retval = tmpdata[1]
		print(params, ' superlu memory: ', retval)

	return [retval] 
	
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

	# mmax = args.mmax
	# nmax = args.nmax
	ntask = args.ntask
	nodes = args.nodes
	cores = args.cores
	nprocmin_pernode = args.nprocmin_pernode
	machine = args.machine
	optimization = args.optimization
	nruns = args.nruns
	truns = args.truns
	obj = args.obj
	# JOBID = args.jobid
	
	TUNER_NAME = args.optimization
	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME
	
	
	nprocmax = nodes*cores  # YL: there is one proc doing spawning, so nodes*cores should be at least 2


	# matrices = ["big.rua", "g4.rua", "g20.rua"]
	# matrices = ["Si2.bin", "SiH4.bin", "SiNa.bin", "Na5.bin", "benzene.bin", "Si10H16.bin", "Si5H12.bin", "SiO.bin", "Ga3As3H12.bin","H2O.bin"]
	matrices = ["big.rua","Si2.bin", "SiH4.bin", "SiNa.bin", "Na5.bin", "benzene.bin", "Si10H16.bin", "Si5H12.bin", "SiO.bin", "Ga3As3H12.bin", "GaAsH6.bin", "H2O.bin"]

	# Task parameters
	matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")

	# Input parameters
	COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
	LOOKAHEAD = Integer     (5, 20, transform="normalize", name="LOOKAHEAD")
	nprows    = Integer     (1, nprocmax, transform="normalize", name="nprows")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
	NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
	NREL      = Integer     (10, 40, transform="normalize", name="NREL")	
	result   = Real        (float("-Inf") , float("Inf"),name="r")
	IS = Space([matrix])
	PS = Space([COLPERM, LOOKAHEAD, npernode, nprows, NSUP, NREL])
	OS = Space([result])

	constraints = {"cst1" : cst1, "cst2" : cst2}
	models = {}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

	target=obj


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
	options['model_class '] = 'Model_LCM' # 'Model_LCM'
	options['verbose'] = False

	options.validate(computer = computer)
	


	""" Intialize the tuner with existing data stored as last check point"""
	try:
		data = pickle.load(open('Data_SLU_nodes_%d_cores_%d_nprocmin_pernode_%d_tasks_%s_machine_%s.pkl' % (nodes, cores, nprocmin_pernode, matrices, machine), 'rb'))
		giventask = data.I
	except (OSError, IOError) as e:
		data = Data(problem)
		giventask = [[np.random.choice(matrices,size=1)[0]] for i in range(ntask)]


	# """ Building MLA with the given list of tasks """
	giventask = [["big.rua"]]		
	# giventask = [["Si2.bin"]]	
	# giventask = [["Si2.bin"],["SiH4.bin"], ["SiNa.bin"], ["Na5.bin"], ["benzene.bin"], ["Si10H16.bin"], ["Si5H12.bin"]]	
	data = Data(problem)



	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = nruns
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
		print("stats: ", stats)


		""" Dump the data to file as a new check point """
		pickle.dump(data, open('Data_SLU_nodes_%d_cores_%d_nprocmin_pernode_%d_tasks_%s_machine_%s.pkl' % (nodes, cores, nprocmin_pernode, matrices, machine), 'wb'))

		""" Dump the tuner to file for TLA use """
		pickle.dump(gt, open('MLA_SLU_nodes_%d_cores_%d_nprocmin_pernode_%d_tasks_%s_machine_%s.pkl' % (nodes, cores, nprocmin_pernode, matrices, machine), 'wb'))

		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))



	if(TUNER_NAME=='opentuner'):
		NI = ntask
		NS = nruns
		(data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
		print("stats: ", stats)

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

	if(TUNER_NAME=='hpbandster'):
		NI = ntask
		NS = nruns
		(data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
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
	parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
	parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nruns', type=int, help='Number of runs per task')
	parser.add_argument('-truns', type=int, help='Time of runs')
	# Experiment related arguments
	parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)
	parser.add_argument('-stepid', type=int, default=-1, help='step ID')
	parser.add_argument('-phase', type=int, default=0, help='phase')
	parser.add_argument('-obj', type=str, default='time', help='Tuning objective (time or memory)')

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
