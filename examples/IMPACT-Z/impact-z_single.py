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



from GPTune.gptune import * # import all



from autotune.problem import *
from autotune.space import *
from autotune.search import *

from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
    
	######################################### 
	##### constants defined in TuningProblem
	nodes = point['nodes']
	cores = point['cores']	
	#########################################

	quad1 = point['quad1']
	quad2 = point['quad2']
	quad3 = point['quad3']
	quad4 = point['quad4']
	quad5 = point['quad5']
	inputfile = point['inputfile']
	controlfile = point['controlfile']

	nproc = nodes*cores
	nproc = 2**(math.floor(math.log(nproc, 2)))
	# nproc =16 # hardcoded now, nproc=32 will make the objective function slightly different ... 
	nthreads = 1
	npernode = cores

	params = [inputfile, controlfile, 'quad1', quad1, 'quad2', quad2, 'quad3', quad3, 'quad4', quad4, 'quad5', quad5]
	os.system("cp "+inputfile+" ./ImpactZ0.in")
	os.system("cp "+controlfile+" ./matchquad.in")

	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	# envstr+= 'NREL=%d\n' %(NREL)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works


	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "./ImpactZexe-mpi", 'args', ['%s'%(quad1), '%s'%(quad2), '%s'%(quad3), '%s'%(quad4), '%s'%(quad5)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
	comm = MPI.COMM_SELF.Spawn("./ImpactZexe-mpi", args=['%s'%(quad1), '%s'%(quad2), '%s'%(quad3), '%s'%(quad4), '%s'%(quad5)], maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """
	tmpdata = np.array([0, 0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MIN,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()
	   
	retval = tmpdata[0]
	print(params, ' impact-z objective: ', retval)

	return [retval] 

def input_var(x):
    # return np.fabs(np.ones([x.shape[0],x.shape[1]]))
    return np.fabs(x)*0.05
    # return np.fabs(x)*0
    # return np.fabs(np.random.rand(x.shape[0],x.shape[1]))

			
def main():

	# Parse command line arguments
	args   = parse_args()

	# Extract arguments
	ntask = args.ntask
	optimization = args.optimization
	nrun = args.nrun
	TUNER_NAME = args.optimization
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME

	os.system("cp ./IMPACT-Z/build/ImpactZexe-mpi .")
	
	nprocmax = nodes*cores
	inputfiles = ["ImpactZ.in_test1","ImpactZ.in_test2"]
	controlfiles = ["matchquad.in_test1","matchquad.in_test2"]
	# Task parameters
	inputfile    = Categoricalnorm (inputfiles, transform="onehot", name="inputfile")
	controlfile    = Categoricalnorm (controlfiles, transform="onehot", name="controlfile")
	# Input parameters
	# we know that XX = x00*(1+quad) has range [-50,50], so adjust range of quad accordingly
	file1 = open('matchquad.in_test1', 'r')
	Lines = file1.readlines()
	npara = int(Lines[0].split()[0])
	res = [i for i in Lines[-1].split()]
	b1 = [-50.0/float(res[i])-1.0 for i in range(npara)]
	b2 = [50.0/float(res[i])-1.0 for i in range(npara)]
	lb = [min(b1[i],b2[i]) for i in range(npara)]
	ub = [max(b1[i],b2[i]) for i in range(npara)]
	
	# quad1 = Real     (lb[0], ub[0], transform="normalize", name="quad1")
	# quad2 = Real     (lb[1], ub[1], transform="normalize", name="quad2")
	# quad3 = Real     (lb[2], ub[2], transform="normalize", name="quad3")
	# quad4 = Real     (lb[3], ub[3], transform="normalize", name="quad4")
	# quad5 = Real     (lb[4], ub[4], transform="normalize", name="quad5")			
	
	quad1 = Real     (-0.06, 0.06, transform="normalize", name="quad1")
	quad2 = Real     (-0.06, 0.06, transform="normalize", name="quad2")
	quad3 = Real     (-0.06, 0.06, transform="normalize", name="quad3")
	quad4 = Real     (-0.06, 0.06, transform="normalize", name="quad4")
	quad5 = Real     (-0.06, 0.06, transform="normalize", name="quad5")			
	


	# Output parameters
	mismatch   = Real        (float("-Inf") , float("Inf"),name="mismatch")
	IS = Space([inputfile,controlfile])
	PS = Space([quad1, quad2, quad3, quad4, quad5])
	OS = Space([mismatch])
	constraints = {}
	models = {}
	constants={"nodes":nodes,"cores":cores}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

	
	problem = TuningProblem(IS, PS, OS, objectives, constraints, models=None, constants=constants, input_var=input_var)
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
	options['model_kern'] = 'WGP'
	options['verbose'] = False
	# options['search_pop_size'] = 10000
	options['sample_class'] = 'SampleOpenTURNS'
	options.validate(computer = computer)

	# """ Building MLA with the given list of tasks """
	# giventask = [[np.random.choice(matrices,size=1)[0]] for i in range(ntask)]
	giventask = [["ImpactZ.in_test1","matchquad.in_test1"]]
	# giventask = [["big.rua"]]	
	data = Data(problem)
	Pdefault = [0, 0, 0, 0, 0]
	data.P = [[Pdefault]] * ntask

	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = nrun
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=int(NS/2))
		print("stats: ", stats)

		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    inputfile:%s controlfile:%s"%(data.I[tid][0],data.I[tid][1]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
			subtracted = [max(1e-5,abs(element1 - element2)/2.0) for (element1, element2) in zip(data.P[tid][np.argmin(data.O[tid])], Pdefault)]
			newmin = [element1 - element2 for (element1, element2) in zip(data.P[tid][np.argmin(data.O[tid])],subtracted)]
			newmax = [element1 + element2 for (element1, element2) in zip(data.P[tid][np.argmin(data.O[tid])],subtracted)]
			print("    new Pdefault:", data.P[tid][np.argmin(data.O[tid])])
			print("    new search range xmin:", newmin)
			print("    new search range xmax:", newmax)


	if(TUNER_NAME=='opentuner'):
		NI = len(giventask)
		NS = nrun
		(data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
		print("stats: ", stats)

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    inputfile:%s controlfile:%s"%(data.I[tid][0],data.I[tid][1]))
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
			print("    inputfile:%s controlfile:%s"%(data.I[tid][0],data.I[tid][1]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))




def parse_args():

	parser = argparse.ArgumentParser()
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')
	
	args   = parser.parse_args()
	return args

if __name__ == "__main__":
	main()
