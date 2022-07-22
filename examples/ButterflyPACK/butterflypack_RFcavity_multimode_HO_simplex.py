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
mpirun -n 1 python butterflypack_RFcavity_multimode.py -nthreads 1 -ntask 20 -nrun 800 -optimization GPTune

where:
	-nthreads is the number of OMP threads in the application run
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
import copy
import time

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
from csv import writer, reader


def read_validdata_onemode(model,nth,freq_check):
	file =open(model+'_order_%s_Nmodes.txt'%(order),'r')
	Lines = file.readlines()
	Nmode = int(Lines[0].strip())
	file.close()
	file =open(model+'_order_%s_freq_history.txt'%(order),'r')
	Lines = file.readlines()
	Nsample = int(Lines[0].strip())
	file.close()

	dict1={}
	mm=nth
	filename=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
	# print(mm,filename)
	file =open(filename,'r')
	Lines = file.readlines()
	for nn in range(len(Lines)): 
		freq =int(round(float(Lines[nn].split()[0])))  # rounding for the ease of looking up data. Note that the precision of the data is much higher than those used in the GPTune experiments
		eigval =float(Lines[nn].split()[1])
		# print(freq,eigval)
		dict1[freq]=eigval
	# print(dict1)
	file.close()

	
	validfreq = freq_check in dict1
	validO = 1e0 # assign a large value if not existing in file
	if(validfreq):
		validO=dict1[freq_check]

	return (validfreq,validO)



def read_validdata(model):
	file =open(model+'_order_%s_Nmodes.txt'%(order),'r')
	Lines = file.readlines()
	Nmode = int(Lines[0].strip())
	file.close()
	file =open(model+'_order_%s_freq_history.txt'%(order),'r')
	Lines = file.readlines()
	Nsample = int(Lines[0].strip())
	file.close()
	Pall=[]
	Oall=[]
	for nth in range(Nmode):
		P=[]
		O=[]
		mm=nth
		filename=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
		# print(mm,filename)
		file =open(filename,'r')
		Lines = file.readlines()
		for nn in range(len(Lines)): 
			freq =int(round(float(Lines[nn].split()[0]))) # rounding for the ease of looking up data. Note that the precision of the data is much higher than those used in the GPTune experiments 
			eigval =float(Lines[nn].split()[1])
			# print(freq,eigval)

			P.append([freq/1e5])
			O.append([eigval])

		file.close()

		Pall.append(P)
		Oall.append(O)
	# print(P)
	# print(O)

	return (Pall,Oall)

################################################################################
def objectives(point, nodes, cores, nthreads, model, nth):                  # should always use this name for user-defined objective function

	
	postprocess=0
	baca_batch=64
	knn=0
	verbosity=1
	norm_thresh=500
	freq = point[0]*1e5
	# freq = 22281*1e5
	freq_int =int(round(freq)) # rounding for the ease of looking up data. Note that the precision of the data is much higher than those used in the GPTune experiments

	nproc     = nodes*cores/nthreads
	npernode =  math.ceil(float(cores)/nthreads) 

	params = [model, 'freq', freq]

	BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/TDFDIE_HO/FDIE_HO_openmpi")
	RUNDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/TDFDIE_HO/FDIE_HO_openmpi")
	os.system("cp %s/fdmom_eigen ."%(BINDIR))
	os.system("cp %s/%s.inp ."%(RUNDIR,model))
	os.system("cp %s/materials.inp ."%(RUNDIR))
	os.system("cp %s/inputs_fd.inp ."%(RUNDIR))
	

	TUNER_NAME = os.environ['TUNER_NAME']
	
	""" pass some parameters through environment variables """	
	info = MPI.Info.Create()
	envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
	info.Set('env',envstr)
	info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    
   
    ##### YL: note that '--noport', '1' as the port issue still remains to be fixed. 

	""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
	comm = MPI.COMM_SELF.Spawn("%s/fdmom_eigen"%(RUNDIR), args=['-quant', '--model', '%s'%(model), '--freq', '%s'%(freq),'--si', '1', '--noport', '1', '--exact_mapping', '1', '--which', 'LM','--norm_thresh','%s'%(norm_thresh),'--ordbasis','%s'%(order),'--nev', '20', '--postprocess', '%s'%(postprocess), '-option', '--tol_comp', '1d-4','--reclr_leaf','5','--lrlevel', '0', '--xyzsort', '2','--nmin_leaf', '100','--format', '1','--sample_para','2d0','--baca_batch','%s'%(baca_batch),'--knn','%s'%(knn),'--level_check','100','--verbosity', '%s'%(verbosity)], maxprocs=nproc,info=info)

	""" gather the return value using the inter-communicator """							
	tmpdata = np.array([0, 0],dtype=np.float64)
	comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
	comm.Disconnect()	
	# read the file to see if a new valid sample has been generated
	(validfreq,retval) = read_validdata_onemode(model,nth,freq_int)

	with open('history.csv', 'a') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow([point[0],retval])
		f_object.close()
	return retval


def readdata(model):
	file =open(model+'_order_%s_Nmodes.txt'%(order),'r')
	Lines = file.readlines()
	Nmode = int(Lines[0].strip())
	file.close()
	file =open(model+'_order_%s_freq_history.txt'%(order),'r')
	Lines = file.readlines()
	Nsample = int(Lines[0].strip())
	file.close()
	dict={}
	for nn in range(len(Lines)-1): 
		freq =int(round(float(Lines[nn+1].strip())/1e5)) 
		# print(freq)
		dict[freq]=1e0

	Pall=[]
	Oall=[]
	for mm in range(Nmode):
		filename=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
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
	global order


	# Parse command line arguments

	args   = parse_args()

	# Extract arguments

	ntask = args.ntask
	nthreads = args.nthreads
	optimization = args.optimization
	nrun = args.nrun
	order = args.order
	
	
	TUNER_NAME = args.optimization	
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))


	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME
	

	


	# """ Building MLA with the given list of tasks """	
	# giventask = [["pillbox_4000"]]		
	# giventask = [["pillbox_1000"]]		
	# giventask = [["rfq_mirror_50K_feko"]]		
	# giventask = [["cavity_5cell_30K_feko"]]		
	giventask = [["cavity_rec_17K_feko"]]		
	# giventask = [["cavity_wakefield_4K_feko"]]		

	if(TUNER_NAME=='SIMPLEX'):
		t3 = time.time_ns()
		(Pall,Oall)=read_validdata(giventask[0][0])
		Nmode = len(Pall)
		nth=0
		while nth<Nmode:
			P = Pall[nth]
			O = Oall[nth]
			print(nth, ' mode:')
			print(P)
			print(O)

			os.system("rm -rf history.csv")

			X=np.array(P)	
			Pdefault = np.asarray(X[np.argmin(np.asarray(O))])

			print('min ', np.amin(X),' max ', np.amax(X), 'mean ', np.mean(X), 'best ', X[np.argmin(O)])

			bounds_constraint = [(np.amin(X)*0.999,np.amax(X)*1.001)]
			sol = sp.optimize.minimize(objectives, Pdefault, args=(nodes, cores, nthreads, giventask[0][0], nth), method='Nelder-Mead', options={'verbose': 1, 'maxfev': nrun, 'xatol': 0.0000001, 'fatol': 0.0000001}, bounds=bounds_constraint)    

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
					Ps.append(float(row[0]))
					Os.append(float(row[1]))
				Os=np.asarray(Os)	
				print("    Simplex refined mode ", nth)
				print("    bounds ", bounds_constraint)
				print("    Ps ", Ps)
				print("    Os ", Os.tolist())	
				print('    Popt ', sol.x, 'Oopt ', sol.fun, 'nth ', np.argmin(Os))	


			# reload the data and Nmode as they may have been updated by other modes
			(Pall,Oall)=read_validdata(giventask[0][0])
			Nmode = len(Pall)
			nth = nth+1


def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nthreads', type=int, default=1,help='Number of OMP threads for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')
	parser.add_argument('-order', type=int, default=0, help='order of the FDIE code')

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
