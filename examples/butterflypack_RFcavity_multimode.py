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
import copy
import time

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
import pygmo as pg
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
	postprocess=0
	baca_batch=64
	knn=0
	verbosity=-1
	norm_thresh=500
	model = point['model']
	freq = point['freq']*1e5
	# freq = 22281*1e5

	nproc     = npernode*nodes
	# nthreads  =1 
	# npernode =  math.ceil(float(nproc)/nodes)  
	nthreads = int(cores / npernode)

	params = [model, 'freq', freq]
	RUNDIR = "/project/projectdirs/m2957/liuyangz/my_research/ButterflyPACK_RF_cavity/build/EXAMPLE"
	INPUTDIR = "/project/projectdirs/m2957/liuyangz/my_research/ButterflyPACK_hss_factor_acc/EXAMPLE/EM3D_DATA/preprocessor_3dmesh"
	TUNER_NAME = os.environ['TUNER_NAME']
	
	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	comm = MPI.COMM_SELF.Spawn("%s/ie3dporteigen"%(RUNDIR), args=['-quant', '--data_dir', '%s/%s'%(INPUTDIR,model), '--model', '%s'%(model), '--freq', '%s'%(freq),'--si', '1', '--which', 'LM','--norm_thresh','%s'%(norm_thresh),'--nev', '20', '--postprocess', '%s'%(postprocess), '--cmmode', '0','-option', '--tol_comp', '1d-4','--lrlevel', '0', '--xyzsort', '2','--nmin_leaf', '100','--format', '1','--sample_para','2d0','--baca_batch','%s'%(baca_batch),'--knn','%s'%(knn),'--verbosity', '%s'%(verbosity)], maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator """							
	tmpdata = np.array([0, 0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	
	# if(tmpdata[1]<100):  # small 1-norm of the eigenvector means this is a false resonance
	# 	tmpdata[0]=1e2
	# print(params, '[ abs of eigval, 1-norm of eigvec ]:', tmpdata)

	# return [tmpdata[0]] 
	
	return [0] # return a dummy value, the true vale will be read from file by readdata



def readdata(model):
	file =open(model+'_Nmodes.txt','r')
	Lines = file.readlines()
	Nmode = int(Lines[0].strip())
	file.close()
	file =open(model+'_freq_history.txt','r')
	Lines = file.readlines()
	Nsample = int(Lines[0].strip())
	file.close()
	dict={}
	for nn in range(len(Lines)-1): 
		freq =int(round(float(Lines[nn+1].strip())/1e5)) 
		# print(freq)
		dict[freq]=1e2

	Pall=[]
	Oall=[]
	for mm in range(Nmode):
		filename=model+'_EigVals_'+str(mm+1)+'.out'
		dict1=copy.deepcopy(dict)
		# print(mm,filename)
		file =open(filename,'r')
		Lines = file.readlines()
		for nn in range(len(Lines)): 
			freq =int(round(float(Lines[nn].split()[0])/1e5)) 
			eigval =float(Lines[nn].split()[1])
			# print(freq,eigval)
			dict1[freq]=min(dict1[freq],eigval)
		# print(dict1)
		file.close()
		P=[]
		O=[]
		for nn in range(len(dict1)):
			P.append([list(dict1.keys())[nn]])
			O.append([list(dict1.values())[nn]])
		Pall.append(P)
		Oall.append(O)
		# print(P)
		# print(O)

	return (Pall,Oall)


	
def main():

	global ROOTDIR
	global nodes
	global cores
	global target
	global nprocmax
	global nprocmin
	global npernode

	
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
	# JOBID = args.jobid
	
	TUNER_NAME = args.optimization
	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME
	
	
	# nprocmax = nodes*cores-1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
	# nprocmin = min(nodes*nprocmin_pernode,nprocmax-1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space
	npernode = nprocmin_pernode

	

	# Task parameters
	geomodels = ["cavity_5cell_30K_feko","pillbox_4000","pillbox_1000","cavity_wakefield_4K_feko","cavity_rec_5K_feko","cavity_rec_17K_feko"]
	# geomodels = ["cavity_wakefield_4K_feko"]
	model    = Categoricalnorm (geomodels, transform="onehot", name="model")


	# Input parameters  # the frequency resolution is 100Khz
	# freq      = Integer     (22000, 23500, transform="normalize", name="freq")
	# freq      = Integer     (15000, 23500, transform="normalize", name="freq")
	# freq      = Integer     (19300, 22300, transform="normalize", name="freq")
	freq      = Integer     (15000, 40000, transform="normalize", name="freq")
	# freq      = Integer     (6320, 6430, transform="normalize", name="freq")
	# freq      = Integer     (21000, 22800, transform="normalize", name="freq")
	# freq      = Integer     (11400, 12000, transform="normalize", name="freq")
	# freq      = Integer     (500, 900, transform="normalize", name="freq")
	result1   = Real        (float("-Inf") , float("Inf"),name="r1")
	result2   = Real        (float("-Inf") , float("Inf"),name="r2")
	
	IS = Space([model])
	PS = Space([freq])
	# OS = Space([result1,result2])
	OS = Space([result1])

	constraints = {}
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
	options['model_class '] = 'Model_LCM' # 'Model_GPy_LCM'
	options['verbose'] = False

	# options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso' 
	# options['search_pop_size'] = 1000 # 1000
	# options['search_gen'] = 10

	options.validate(computer = computer)
	


	# """ Intialize the tuner with existing data stored as last check point"""
	# try:
	# 	data = pickle.load(open('Data_nodes_%d_cores_%d_nprocmin_pernode_%d_tasks_%s_machine_%s.pkl' % (nodes, cores, nprocmin_pernode, geomodels, machine), 'rb'))
	# 	giventask = data.I
	# except (OSError, IOError) as e:
	# 	data = Data(problem)
	# 	giventask = [[np.random.choice(geomodels,size=1)[0]] for i in range(ntask)]


	# """ Building MLA with the given list of tasks """
	# giventask = [["big.rua"]]		
	# giventask = [["pillbox_4000"]]		
	# giventask = [["cavity_5cell_30K_feko"]]		
	giventask = [["cavity_rec_17K_feko"]]		
	# giventask = [["cavity_wakefield_4K_feko"]]		
	



	if(TUNER_NAME=='GPTune'):
		t3 = time.time_ns()
		data = Data(problem)
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = max(nruns//2, 1)
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS)
		
		(Pall,Oall) = readdata(giventask[0][0])
	
		print("Pall: ", Pall)
		print("Oall: ", Oall)

		try:
			file =open(giventask[0][0]+'_Nmodes.txt','r')
			Lines = file.readlines()
			Nmode = int(Lines[0].strip())
			file.close()			
		except IOError:
			Nmode = 0
			print("no mode found in the intial samples")
		
		for nn in range(NS):
			mm=0
			while mm<Nmode:
				data = Data(problem)
				data.P=[Pall[mm]]
				data.O=[np.array(Oall[mm])]
				gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
				
				NI = len(giventask)
				(data, model, stats) = gt.MLA(NS=len(data.P[0])+1, NI=NI, Igiven=giventask, NS1=len(data.P[0]))
				
				(Pall,Oall) = readdata(giventask[0][0])

				file =open(giventask[0][0]+'_Nmodes.txt','r')
				Lines = file.readlines()
				Nmode = int(Lines[0].strip())
				file.close()
				mm +=1


		""" Print all input and parameter samples """	
		for mm in range(Nmode):
			print("mode: %d"%(mm))
			print("    geometry:%s"%(giventask[0][0]))
			print("    Ps ", Pall[mm])
			
			OL=np.asarray([o[0] for o in Oall[mm]], dtype=np.float64)
			np.set_printoptions(suppress=False,precision=8)	
			print("    Os ", OL)
			print('    Popt ', Pall[mm][np.argmin(Oall[mm])], 'Oopt ', min(Oall[mm])[0], 'nth ', np.argmin(Oall[mm]))
		t4 = time.time_ns()
		print("Total time: ", (t4-t3)/1e9)
	

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

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
