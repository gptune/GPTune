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
from csv import writer, reader
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all



from autotune.problem import *
from autotune.space import *
from autotune.search import *

from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point, *args):                  
    
	######################################### 
	##### constants defined in TuningProblem
	nodes = 1
	cores = 32
	#########################################

	quad1 = point[0]
	quad2 = point[1]
	quad3 = point[2]
	quad4 = point[3]
	quad5 = point[4]
	inputfile = "ImpactZ.in_test1"
	controlfile = "matchquad.in_test1"


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

	with open('history.csv', 'a') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow([quad1,quad2,quad3,quad4,quad5,retval])
		f_object.close()

	return retval

def compareVersion(version1, version2):
	versions1 = [int(v) for v in version1.split(".")]
	versions2 = [int(v) for v in version2.split(".")]
	for i in range(max(len(versions1),len(versions2))):
		v1 = versions1[i] if i < len(versions1) else 0
		v2 = versions2[i] if i < len(versions2) else 0
		if v1 > v2:
			return 1
		elif v1 <v2:
			return -1
	return 0
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
	# inputfiles = ["ImpactZ.in_test1","ImpactZ.in_test2"]
	# controlfiles = ["matchquad.in_test1","matchquad.in_test2"]
	# # Task parameters
	# inputfile    = Categoricalnorm (inputfiles, transform="onehot", name="inputfile")
	# controlfile    = Categoricalnorm (controlfiles, transform="onehot", name="controlfile")
	# # Input parameters
	# # we know that XX = x00*(1+quad) has range [-50,50], so adjust range of quad accordingly
	# file1 = open('matchquad.in_test1', 'r')
	# Lines = file1.readlines()
	# npara = int(Lines[0].split()[0])
	# res = [i for i in Lines[-1].split()]
	# b1 = [-50.0/float(res[i])-1.0 for i in range(npara)]
	# b2 = [50.0/float(res[i])-1.0 for i in range(npara)]
	# lb = [min(b1[i],b2[i]) for i in range(npara)]
	# ub = [max(b1[i],b2[i]) for i in range(npara)]
	
	# quad1 = Real     (lb[0], ub[0], transform="normalize", name="quad1")
	# quad2 = Real     (lb[1], ub[1], transform="normalize", name="quad2")
	# quad3 = Real     (lb[2], ub[2], transform="normalize", name="quad3")
	# quad4 = Real     (lb[3], ub[3], transform="normalize", name="quad4")
	# quad5 = Real     (lb[4], ub[4], transform="normalize", name="quad5")			
	
	# quad1 = Real     (-0.05, 0.05, transform="normalize", name="quad1")
	# quad2 = Real     (-0.05, 0.05, transform="normalize", name="quad2")
	# quad3 = Real     (-0.05, 0.05, transform="normalize", name="quad3")
	# quad4 = Real     (-0.05, 0.05, transform="normalize", name="quad4")
	# quad5 = Real     (-0.05, 0.05, transform="normalize", name="quad5")			
	


	## Output parameters
	# mismatch   = Real        (float("-Inf") , float("Inf"),name="mismatch")
	# IS = Space([inputfile,controlfile])
	# PS = Space([quad1, quad2, quad3, quad4, quad5])
	# OS = Space([mismatch])
	# constraints = {}
	# models = {}
	# constants={"nodes":nodes,"cores":cores}

	# """ Print all input and parameter samples """	
	# # print(IS, PS, OS, constraints, models)

	Pdefault = np.asarray([0,0,0,0,0])
	if(TUNER_NAME=='SIMPLEX'):
		if(compareVersion(sp.__version__,'1.7.0')==-1):
			raise Exception(f"Simplex require scipy version >=1.7.0 to support bounds")
		os.system("rm -rf history.csv")

		lw = [-1]*Pdefault.shape[0]
		up = [1]*Pdefault.shape[0]
		bounds_constraint = sp.optimize.Bounds(lw, up)
		sol = sp.optimize.minimize(objectives, Pdefault, method='Nelder-Mead', options={'verbose': 1, 'maxfev': nrun, 'xatol': 0.0000001, 'fatol': 0.0000001}, bounds=bounds_constraint)    

		print('x      : ', sol.x)
		print('fun      : ', sol.fun)
		#print('hess_inv : ', sol.hess_inv)
		#print('jac      : ', jac)
		print('message  : ', sol.message)
		print('nfev     : ', sol.nfev)
		print('nit      : ', sol.nit)
		print('status   : ', sol.status)
		print('success  : ', sol.success)
		with open('history.csv', mode='r') as f_object:
			csv_reader = reader(f_object, delimiter=',')
			Ps=[]
			Os=[]
			for row in csv_reader:
				Ps.append([float(a) for a in row[0:5]])
				Os.append(float(row[5]))
			Os=np.asarray(Os)	
			print("    Ps ", Ps)
			print("    Os ", Os.tolist())	
			print('    Popt ', sol.x, 'Oopt ', sol.fun, 'nth ', np.argmin(Os))	
			X=np.array(Ps)				
			brs = NearestNeighbors(n_neighbors=min(50,len(Ps)), algorithm='auto').fit(X)	
			indices = brs.kneighbors(X[np.argmin(Os)].reshape(1, -1),return_distance=False)
			# print(X[indices])
			print(Os[indices])
			print("    new Pdefault:", sol.x)
			print("    new search range xmin:", np.amin(X[indices],1))
			print("    new search range xmax:", np.amax(X[indices],1))



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
