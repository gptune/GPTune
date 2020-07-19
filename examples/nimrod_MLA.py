
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
	mx = point['mx']
	my = point['my']
	lphi = point['lphi']
	nstep = point['nstep']
	# nprows = 2**point['nprows']
	# nproc = 2**point['nproc']
	# nproc = 32
	NSUP = point['NSUP']
	NREL = point['NREL']
	nbx = point['nbx']
	nby = point['nby']
	# nblock     = int(nprocmax/nproc)
	# npcols     = int(nproc/ nprows)
	params = ['mx',mx,'my',my,'lphi',lphi,'nstep',nstep,'ROWPERM', ROWPERM, 'COLPERM', COLPERM, 'NSUP', NSUP, 'NREL', NREL, 'nbx', nbx, 'nby', nby]

	# INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
	TUNER_NAME = os.environ['TUNER_NAME']

	nthreads   = 1


	""" pass some parameters through environment variables """	


	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=1\n'
	envstr+= 'NREL=%d\n' %(NREL)   
	envstr+= 'NSUP=%d\n' %(NSUP)   
	info.Set('env',envstr)


	fin = open("./nimrod_template.in","rt")
	fout = open("./nimrod.in","wt")

	for line in fin:
		#read replace the string and write to output file
		if(line.find("iopts(3)")!=-1):
			fout.write("iopts(3)= %s\n"%(ROWPERM))
		elif(line.find("iopts(4)")!=-1):
			fout.write("iopts(4)= %s\n"%(COLPERM))    
		elif(line.find("lphi")!=-1):
			fout.write("lphi= %s\n"%(lphi))    
		elif(line.find("nlayers")!=-1):
			fout.write("nlayers= %s\n"%(int(np.floor(2**lphi/3.0)+1)))  	
		elif(line.find("mx")!=-1):
			fout.write("mx= %s\n"%(2**mx)) 
		elif(line.find("nstep")!=-1):
			fout.write("nstep= %s\n"%(nstep))  			 
		elif(line.find("my")!=-1):
			fout.write("my= %s\n"%(2**my))   
		elif(line.find("nxbl")!=-1):
			fout.write("nxbl= %s\n"%(int(2**mx/2**nbx)))  
		elif(line.find("nybl")!=-1):
			fout.write("nybl= %s\n"%(int(2**my/2**nby)))  									  						        
		else:
			fout.write(line)
	#close input and output files
	fin.close()
	fout.close()


	nlayers=int(np.floor(2**lphi/3.0)+1)
	nproc = int(nprocmax/nlayers)*nlayers
	if(nprocmax<nlayers):
		print('nprocmax', nprocmax, 'nlayers', nlayers, 'decrease lphi!')
		raise Exception("nprocmax<nlayers")

	if(nproc>int(2**mx/2**nbx)*int(2**my/2**nby)*int(np.floor(2**lphi/3.0)+1)): # nproc <= nlayers*nxbl*nybl
		nproc = int(2**mx/2**nbx)*int(2**my/2**nby)*int(np.floor(2**lphi/3.0)+1) 

	os.system("./nimset")

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	print('exec', "./nimrod_spawn", 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL))
	comm = MPI.COMM_SELF.Spawn("./nimrod_spawn", maxprocs=nproc,info=info)


	# retval=1.0


	""" gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """																	
	tmpdata = np.array([0,0,0,0,0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	
	retval = tmpdata[0]
	print(params, ' nimrod time -- loop:', tmpdata[0],'slu: ', tmpdata[1],'factor: ', tmpdata[2], 'iter: ', tmpdata[3], 'total: ', tmpdata[4])

	return retval 
	
	
def main():

	global ROOTDIR
	global nodes
	global cores
	global target
	global nprocmax

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


	nprocmax = nodes*cores
	# nprocmax = 256

	# Input parameters
	ROWPERM   = Categoricalnorm (['1', '2'], transform="onehot", name="ROWPERM")
	COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
	# nprows    = Integer     (0, 5, transform="normalize", name="nprows")
	# nproc    = Integer     (5, 6, transform="normalize", name="nproc")
	NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
	NREL      = Integer     (10, 40, transform="normalize", name="NREL")
	nbx      = Integer     (1, 3, transform="normalize", name="nbx")	
	nby      = Integer     (1, 3, transform="normalize", name="nby")	

	result   = Real        (float("-Inf") , float("Inf"), transform="normalize", name="r")

	nstep      = Integer     (3, 15, transform="normalize", name="nstep")
	lphi      = Integer     (2, 3, transform="normalize", name="lphi")
	mx      = Integer     (5, 6, transform="normalize", name="mx")
	my      = Integer     (7, 8, transform="normalize", name="my")
	
	IS = Space([mx,my,lphi,nstep])
	# PS = Space([ROWPERM, COLPERM, nprows, nproc, NSUP, NREL])
	PS = Space([ROWPERM, COLPERM, NSUP, NREL, nbx, nby])
	OS = Space([result])
	cst1 = "NSUP >= NREL"
	constraints = {"cst1" : cst1}
	models = {}

	""" Print all input and parameter samples """	
	print(IS, PS, OS, constraints, models)

	BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_haswell_gnu_openmpi/bin")
	RUNDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimrod_input")
	os.system("cp %s/nimrod.in ./nimrod_template.in"%(RUNDIR))
	os.system("cp %s/fluxgrid.in ."%(RUNDIR))
	os.system("cp %s/g163518.03130 ."%(RUNDIR))
	os.system("cp %s/p163518.03130 ."%(RUNDIR))
	os.system("cp %s/nimset ."%(RUNDIR))
	os.system("cp %s/nimrod ./nimrod_spawn"%(BINDIR))


	# target='memory'
	target='time'


	problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
	computer = Computer(nodes = nodes, cores = cores, hosts = None)  

	""" Set and validate options """	
	options = Options()
	options['model_processes'] = 1
	# options['model_threads'] = 1
	options['model_restarts'] = 8
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
	giventask = [[6,8,2,3],[6,8,2,3],[6,8,2,3],[6,8,2,15]]	
	NI = len(giventask)
	NS = nruns
	(data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = max(NS//2,1))
	# (data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = 10)
	print("stats: ",stats)


	""" Print all input and parameter samples """	
	for tid in range(NI):
		print("tid: %d"%(tid))
		print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
		print("    Ps ", data.P[tid])
		print("    Os ", data.O[tid])
		print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


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
