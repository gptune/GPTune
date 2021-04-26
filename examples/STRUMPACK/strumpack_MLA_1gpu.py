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
mpirun -n 1 python strumpack_MLA_1gpu.py -npernode 1 -ntask 20 -nrun 800 

where:
	-npernode is the number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
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
	npernode = point['npernode']	
	
	#########################################
	#     
	# gridsize = point['gridsize']
	matrix = point['matrix']
	sp_reordering_method = point['sp_reordering_method']
	# sp_compression = point['sp_compression']
	# sp_compression1 = sp_compression
	sp_nd_param = point['sp_nd_param']
	sp_gpu_streams = point['sp_gpu_streams']
	# sp_compression_min_sep_size = point['sp_compression_min_sep_size']*1000
	# sp_compression_min_front_size = point['sp_compression_min_front_size']*1000
	# sp_compression_leaf_size = 2**point['sp_compression_leaf_size']
	# sp_compression_rel_tol = 10.0**point['sp_compression_rel_tol']
	
	# sp_compression = 'none'
	extra_str=[' ']	
	# if(sp_compression == 'hss'):
	# 	extra_str=['--hss_rel_tol', '%s'%(sp_compression_rel_tol)]
	# elif(sp_compression == 'blr'):
	# 	extra_str=['--blr_rel_tol', '%s'%(sp_compression_rel_tol)]
	# elif(sp_compression == 'hodlr'):
	# 	extra_str=['--hodlr_rel_tol', '%s'%(sp_compression_rel_tol), '--hodlr_butterfly_levels', '0']
	# elif(sp_compression == 'hodbf'):
	# 	extra_str=['--hodlr_rel_tol', '%s'%(sp_compression_rel_tol), '--hodlr_butterfly_levels', '100']
	# 	sp_compression1 = 'hodlr'
	# elif(sp_compression == 'none'):
	# 	extra_str=[' ']

	if(sp_reordering_method == 'metis'):
		extra_str = extra_str + ['--sp_enable_METIS_NodeNDP']

	nproc = nodes*npernode
	nthreads = 1
	
	params = ['matrix', matrix, 'sp_reordering_method', sp_reordering_method,'sp_nd_param', sp_nd_param, 'sp_gpu_streams', sp_gpu_streams]+extra_str
	RUNDIR = os.path.abspath(__file__ + "/../STRUMPACK/build/examples")
	INPUTDIR = os.path.abspath(__file__ + "/../../SuperLU_DIST/superlu_dist/EXAMPLE/")
	TUNER_NAME = os.environ['TUNER_NAME']
	

	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "%s/testMMdoubleMPIDist"%(RUNDIR), 'args', ['%s/%s'%(INPUTDIR,matrix), '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_nd_param', '%s'%(sp_nd_param), '--sp_gpu_streams', '%s'%(sp_gpu_streams)]+extra_str, 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
	comm = MPI.COMM_SELF.Spawn("%s/testMMdoubleMPIDist"%(RUNDIR), args=['%s/%s'%(INPUTDIR,matrix), '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_nd_param', '%s'%(sp_nd_param), '--sp_gpu_streams', '%s'%(sp_gpu_streams)]+extra_str, maxprocs=nproc,info=info)

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
	npernode = args.npernode
	optimization = args.optimization
	nrun = args.nrun
	
	TUNER_NAME = args.optimization

	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME
	
	
	# nprocmax = nodes*cores-1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
	# nprocmin = min(nodes*nprocmin_pernode,nprocmax-1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space

	# matrices = ["big.rua", "g4.rua", "g20.rua"]
	# matrices = ["Si2.bin", "SiH4.bin", "SiNa.bin", "Na5.bin", "benzene.bin", "Si10H16.bin", "Si5H12.bin", "SiO.bin", "Ga3As3H12.bin","H2O.bin"]
	# matrices = ["Si2.bin", "SiH4.bin", "SiNa.bin", "Na5.bin", "benzene.bin", "Si10H16.bin", "Si5H12.bin", "SiO.bin", "Ga3As3H12.bin", "GaAsH6.bin", "H2O.bin"]

	# Task parameters
	matrices = ["matrix_ACTIVSg10k_AC_00.mtx", "matrix_ACTIVSg70k_AC_00.mtx", "temp_75k.mtx"]
	matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")

	# Input parameters
	sp_reordering_method   = Categoricalnorm (['metis','parmetis','scotch'], transform="onehot", name="sp_reordering_method")
	# sp_reordering_method   = Categoricalnorm (['metis','geometric'], transform="onehot", name="sp_reordering_method")
	# sp_compression   = Categoricalnorm (['none','hss'], transform="onehot", name="sp_compression")
	# sp_compression   = Categoricalnorm (['none','hss','hodlr','hodbf'], transform="onehot", name="sp_compression")
	# sp_compression   = Categoricalnorm (['none','hss','hodlr','hodbf','blr'], transform="onehot", name="sp_compression")
	# npernode     = Integer     (0, 5, transform="normalize", name="npernode")
	sp_nd_param     = Integer     (2, 32, transform="normalize", name="sp_nd_param")
	sp_gpu_streams     = Integer     (1, 8, transform="normalize", name="sp_gpu_streams")
	# sp_compression_min_sep_size     = Integer     (2, 5, transform="normalize", name="sp_compression_min_sep_size")
	# sp_compression_min_front_size     = Integer     (4, 10, transform="normalize", name="sp_compression_min_front_size")
	# sp_compression_leaf_size     = Integer     (5, 9, transform="normalize", name="sp_compression_leaf_size")
	# sp_compression_rel_tol     = Integer(-6, -1, transform="normalize", name="sp_compression_rel_tol")


	result   = Real        (float("-Inf") , float("Inf"),name="r")
	IS = Space([matrix])
	PS = Space([sp_reordering_method,sp_nd_param,sp_gpu_streams])
	OS = Space([result])
	constraints = {}
	models = {}
	constants={"nodes":nodes,"cores":cores,"npernode":npernode}

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
	options['model_class'] = 'Model_GPy_LCM' # 'Model_GPy_LCM'
	options['verbose'] = False

	options.validate(computer = computer)
	
	
	# """ Building MLA with the given list of tasks """
	giventask = [["temp_75k.mtx"]]		
	data = Data(problem)
	
	# the following makes sure the first sample is using default parameters 
	data.I = giventask
	data.P = [[['metis',8,4]]]



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
	# Machine related arguments
	parser.add_argument('-npernode', type=int, default=1,help='Number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
