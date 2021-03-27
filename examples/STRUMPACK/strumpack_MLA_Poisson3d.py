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
mpirun -n 1 python ./strumpack_MLA_Poisson3d.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -optimization GPTune

where:
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
	-optimization is the optimization algorithm: GPTune,opentuner,hpbandster 
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
	#########################################

	gridsize = point['gridsize']
	sp_reordering_method = point['sp_reordering_method']
	sp_compression = point['sp_compression']
	sp_compression1 = sp_compression
	sp_nd_param = point['sp_nd_param']
	sp_compression_min_sep_size = point['sp_compression_min_sep_size']*1000
	sp_compression_min_front_size = point['sp_compression_min_front_size']*1000
	sp_compression_leaf_size = 2**point['sp_compression_leaf_size']
	sp_compression_rel_tol = 10.0**point['sp_compression_rel_tol']
	
	if(sp_compression == 'hss'):
		extra_str=['--hss_rel_tol', '%s'%(sp_compression_rel_tol)]
	elif(sp_compression == 'blr'):
		extra_str=['--blr_rel_tol', '%s'%(sp_compression_rel_tol)]
	elif(sp_compression == 'hodlr'):
		extra_str=['--hodlr_rel_tol', '%s'%(sp_compression_rel_tol), '--hodlr_butterfly_levels', '0']
	elif(sp_compression == 'hodbf'):
		extra_str=['--hodlr_rel_tol', '%s'%(sp_compression_rel_tol), '--hodlr_butterfly_levels', '100']
		sp_compression1 = 'hodlr'
	elif(sp_compression == 'none'):
		extra_str=[' ']

	if(sp_reordering_method == 'metis'):
		extra_str = extra_str + ['--sp_enable_METIS_NodeNDP']

	npernode = 2**point['npernode']
	nproc = nodes*npernode
	nthreads = int(cores / npernode)
	
	params = ['gridsize', gridsize, 'sp_reordering_method', sp_reordering_method,'sp_compression', sp_compression1, 'sp_nd_param', sp_nd_param, 'sp_compression_min_sep_size', sp_compression_min_sep_size, 'sp_compression_min_front_size', sp_compression_min_front_size, 'sp_compression_leaf_size', sp_compression_leaf_size, 'nthreads', nthreads, 'npernode', npernode, 'nproc',nproc]+extra_str
	RUNDIR = os.path.abspath(__file__ + "/../STRUMPACK/build/examples")
	TUNER_NAME = os.environ['TUNER_NAME']
	

	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "%s/testPoisson3dMPIDist"%(RUNDIR), 'args', ['%s'%(gridsize), '1', '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_compression', '%s'%(sp_compression1),'--sp_nd_param', '%s'%(sp_nd_param),'--sp_compression_min_sep_size', '%s'%(sp_compression_min_sep_size),'--sp_compression_min_front_size', '%s'%(sp_compression_min_front_size),'--sp_compression_leaf_size', '%s'%(sp_compression_leaf_size)]+extra_str, 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
	comm = MPI.COMM_SELF.Spawn("%s/testPoisson3dMPIDist"%(RUNDIR), args=['%s'%(gridsize), '1', '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_compression', '%s'%(sp_compression1),'--sp_nd_param', '%s'%(sp_nd_param),'--sp_compression_min_sep_size', '%s'%(sp_compression_min_sep_size),'--sp_compression_min_front_size', '%s'%(sp_compression_min_front_size),'--sp_compression_leaf_size', '%s'%(sp_compression_leaf_size)]+extra_str, maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator """																	
	tmpdata = np.array([0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	

	retval = tmpdata[0]
	print(params, ' strumpack time: ', retval)


	return [retval] 
	
	
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


	# Task parameters
	gridsize = Integer     (30, 100, transform="normalize", name="gridsize")

	# Input parameters
	sp_reordering_method   = Categoricalnorm (['metis','parmetis','geometric'], transform="onehot", name="sp_reordering_method")
	# sp_reordering_method   = Categoricalnorm (['metis','geometric'], transform="onehot", name="sp_reordering_method")
	# sp_compression   = Categoricalnorm (['none','hss'], transform="onehot", name="sp_compression")
	# sp_compression   = Categoricalnorm (['none','hss','hodlr','hodbf'], transform="onehot", name="sp_compression")
	sp_compression   = Categoricalnorm (['none','hss','hodlr','hodbf','blr'], transform="onehot", name="sp_compression")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
	sp_nd_param     = Integer     (8, 32, transform="normalize", name="sp_nd_param")
	sp_compression_min_sep_size     = Integer     (2, 5, transform="normalize", name="sp_compression_min_sep_size")
	sp_compression_min_front_size     = Integer     (4, 10, transform="normalize", name="sp_compression_min_front_size")
	sp_compression_leaf_size     = Integer     (5, 9, transform="normalize", name="sp_compression_leaf_size")
	sp_compression_rel_tol     = Integer(-6, -1, transform="normalize", name="sp_compression_rel_tol")


	result   = Real        (float("-Inf") , float("Inf"),name="r")
	IS = Space([gridsize])
	PS = Space([sp_reordering_method,sp_compression,sp_nd_param,sp_compression_min_sep_size,sp_compression_min_front_size,sp_compression_leaf_size,sp_compression_rel_tol,npernode])
	OS = Space([result])
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

	options.validate(computer = computer)
	
	
	# """ Building MLA with the given list of tasks """
	giventask = [[30]]		
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
			print("    Os ", data.O[tid])
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

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
