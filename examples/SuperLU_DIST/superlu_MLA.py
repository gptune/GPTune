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
mpirun -n 1 python superlu_MLA.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -obj time -tla 0

where:
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
	-obj is the tuning objective: "time" or "memory"
	-tla is whether to perform TLA after MLA
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

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all



from autotune.problem import *
from autotune.space import *
from autotune.search import *

from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
    
	######################################### 
	##### constants defined in TuningProblem
	nodes = point['nodes']
	cores = point['cores']
	target = point['target']	
	#########################################

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
def cst2(npernode,nprows,nodes):
	return nodes * 2**npernode >= nprows
			
def main():

	# Parse command line arguments
	args   = parse_args()

	# Extract arguments
	tla = args.tla
	ntask = args.ntask
	nprocmin_pernode = args.nprocmin_pernode
	optimization = args.optimization
	nrun = args.nrun
	obj = args.obj
	target=obj
	TUNER_NAME = args.optimization
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME


	nprocmax = nodes*cores
	# matrices = ["big.rua", "g4.rua", "g20.rua"]
	# matrices = ["Si2.rb", "SiH4.rb", "SiNa.rb", "Na5.rb", "benzene.rb", "Si10H16.rb", "Si5H12.rb", "SiO.rb", "Ga3As3H12.rb","H2O.rb"]
	# matrices = ["Si2.rb", "SiH4.rb", "SiNa.rb", "Na5.rb", "benzene.rb", "Si10H16.rb", "Si5H12.rb", "SiO.rb", "Ga3As3H12.rb", "GaAsH6.rb", "H2O.rb"]
	matrices = ["big.rua","g20.rua","Si2.bin", "SiH4.bin", "SiNa.bin", "Na5.bin", "benzene.bin", "Si10H16.bin", "Si5H12.bin", "SiO.bin", "Ga3As3H12.bin", "GaAsH6.bin", "H2O.bin"]
	# Task parameters
	matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")
	# Input parameters
	COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
	LOOKAHEAD = Integer     (5, 20, transform="normalize", name="LOOKAHEAD")
	nprows    = Integer     (1, nprocmax, transform="normalize", name="nprows")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
	NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
	NREL      = Integer     (10, 40, transform="normalize", name="NREL")	
	if(target=='time'):			
		result   = Real        (float("-Inf") , float("Inf"),name="time")
	if(target=='memory'):	
		result   = Real        (float("-Inf") , float("Inf"),name="memory")

	IS = Space([matrix])
	PS = Space([COLPERM, LOOKAHEAD, npernode, nprows, NSUP, NREL])
	OS = Space([result])
	constraints = {"cst1" : cst1, "cst2" : cst2}
	models = {}
	constants={"nodes":nodes,"cores":cores,"target":target}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

	
	problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
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
	options['model_class '] = 'Model_LCM' # 'Model_GPy_LCM'
	options['verbose'] = False
	options.validate(computer = computer)

	# """ Building MLA with the given list of tasks """
	# giventask = [[np.random.choice(matrices,size=1)[0]] for i in range(ntask)]
	giventask = [["big.rua"], ["g20.rua"]]
	# giventask = [["big.rua"]]	
	data = Data(problem)


	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = nrun
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
		print("stats: ", stats)

		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

		if(tla==1):
			""" Call TLA for a new task using the constructed LCM model"""    
			newtask = [["big.rua"]]
			# newtask = [["H2O.rb"]]
			(aprxopts,objval,stats) = gt.TLA1(newtask, NS=None)
			print("stats: ",stats)

			""" Print the optimal parameters and function evaluations"""	
			for tid in range(len(newtask)):
				print("new task: %s"%(newtask[tid]))
				print('    predicted Popt: ', aprxopts[tid], ' objval: ',objval[tid]) 	
				


	if(TUNER_NAME=='opentuner'):
		NI = len(giventask)
		NS = nrun
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
		NI = len(giventask)
		NS = nrun
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
	parser.add_argument('-obj', type=str, default='time', help='Tuning objective (time or memory)')	
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')
	parser.add_argument('-tla', type=int, default=0, help='Whether to perform TLA after MLA')

	args   = parser.parse_args()
	return args

if __name__ == "__main__":
	main()
