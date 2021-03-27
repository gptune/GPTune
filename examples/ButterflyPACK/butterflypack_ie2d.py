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
mpirun -n 1 python butterflypack_ie2d.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -optimization GPTune

where:
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
	-optimization is the optimization algorithm: GPTune, hpbandster or opentuner
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
import pygmo as pg
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
    
	######################################### 
	##### constants defined in TuningProblem
	nodes = point['nodes']
	cores = point['cores']	
	#########################################

	model2d = point['model2d']
	nunk = point['nunk']
	wavelength = point['wavelength']

	lrlevel = point['lrlevel']
	xyzsort = point['xyzsort']
	nmin_leaf = 2**point['nmin_leaf']
	npernode = 2**point['npernode']
	nproc = nodes*npernode 
	nthreads = int(cores / npernode)

	params = ['model2d', model2d,'nunk', nunk,'wavelength', wavelength,'lrlevel', lrlevel,'xyzsort', xyzsort,'nmin_leaf', nmin_leaf, 'nproc', nproc]

	RUNDIR = os.path.abspath(__file__ + "/../ButterflyPACK/build/EXAMPLE")
	TUNER_NAME = os.environ['TUNER_NAME']
	


	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "%s/ie2d"%(RUNDIR), 'args', ['-quant', '--model2d', '%s'%(model2d), '--wavelength', '%s'%(wavelength),'--nunk', '%s'%(nunk),'-option', '--tol_comp', '1d-4','--lrlevel', '%s'%(lrlevel),'--xyzsort', '%s'%(xyzsort),'--nmin_leaf', '%s'%(nmin_leaf),'--format', '1','--precon', '3','--sample_para','2d0','--knn','20','--verbosity','1'], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
	comm = MPI.COMM_SELF.Spawn("%s/ie2d"%(RUNDIR), args=['-quant', '--model2d', '%s'%(model2d), '--wavelength', '%s'%(wavelength),'--nunk', '%s'%(nunk),'-option', '--tol_comp', '1d-4','--lrlevel', '%s'%(lrlevel),'--xyzsort', '%s'%(xyzsort),'--nmin_leaf', '%s'%(nmin_leaf),'--format', '1','--precon', '3','--sample_para','2d0','--knn','20','--verbosity','1'], maxprocs=nproc,info=info)


	""" gather the return value using the inter-communicator """							
	tmpdata = np.array([0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	
	print(params, 'time:', tmpdata[0])

	return [tmpdata[0]] 

	
def main():

	
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



	# Task parameters
	model2d 	= Integer     (1, 13, transform="normalize", name="model2d")
	nunk 		= Integer     (2000, 10000000, transform="normalize", name="nunk")
	wavelength  = Real        (0.00001 , 0.02,name="wavelength")


	# Input parameters
	lrlevel   = Categoricalnorm (['0','100'], transform="onehot", name="lrlevel")
	xyzsort   = Categoricalnorm (['0','1','2'], transform="onehot", name="xyzsort")
	nmin_leaf = Integer     (5, 9, transform="normalize", name="nmin_leaf")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")


	result1   = Real        (float("-Inf") , float("Inf"),name="r1")


	IS = Space([model2d,nunk,wavelength])
	PS = Space([lrlevel,xyzsort,nmin_leaf,npernode])
	OS = Space([result1])

	constraints = {}
	models = {}
	constants={"nodes":nodes,"cores":cores}

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

	# options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso' 
	# options['search_pop_size'] = 1000 # 1000
	# options['search_gen'] = 10

	options.validate(computer = computer)
	

	# """ Building MLA with the given list of tasks """	
	# giventask = [[7,100000,0.001]]			
	giventask = [[7,5000,0.02]]			
	# giventask = [[7,2000,0.05]]			
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
			print("    model2d:%d nunk:%d wavelength:%1.6e" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
			print("    Ps ", data.P[tid])
			

			OL=np.asarray([o[0] for o in data.O[tid]], dtype=np.float64)
			np.set_printoptions(suppress=False,precision=8)	
			print("    Os ", OL)
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

			# ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data.O[tid])
			# front = ndf[0]
			# # print('front id: ',front)
			# fopts = data.O[tid][front]
			# xopts = [data.P[tid][i] for i in front]
			# print('    Popts ', xopts)		
			# print('    Oopts ', fopts)

	if(TUNER_NAME=='opentuner'):
		NI = ntask
		NS = nrun
		(data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
		print("stats: ", stats)

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid])
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

	if(TUNER_NAME=='hpbandster'):
		NI = ntask
		NS = nrun
		(data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
		print("stats: ", stats)
		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid])
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))





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
