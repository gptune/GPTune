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

			idxs=[index for (index, item) in enumerate(P) if item == [freq/1e5]]
			if(len(idxs)>0): # if already evaluated, take the minimum 
				O[idxs[0]]=[min(O[idxs[0]][0],freq/1e5)]
			else:
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

	tdplot=0
	noloss=0
	baca_batch=16
	knn=0
	verbosity=1
	freq = point[0]*1e5
	# freq = 22281*1e5
	freq_int =int(round(freq)) # rounding for the ease of looking up data. Note that the precision of the data is much higher than those used in the GPTune experiments

	# read the file to see if the frequency has already been evaluated
	(validfreq,retval) = read_validdata_onemode(model,nth,freq_int)

	if(validfreq and postprocess==0):
		print("skip frequency %s as it has already been evaluated"%(freq))
		retval=retval*0.99 # this tricks scikit-optimize to not terminate early, note that only the csv files (only used by scikit-optimize) use these modified values
	else:
		nproc     = nodes*cores/nthreads
		npernode =  math.ceil(float(cores)/nthreads) 

		params = [model, 'freq', freq]

		BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/TDFDIE_HO/FDIE_HO_openmpi_arbiport")
		RUNDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/TDFDIE_HO/FDIE_HO_openmpi_arbiport")
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
		
		""" use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
		comm = MPI.COMM_SELF.Spawn("%s/fdmom_eigen"%(RUNDIR), args=['-quant', '--model', '%s'%(model), '--freq', '%s'%(freq),'--si', '1', '--noport', '%s'%(noport), '--noloss', '%s'%(noloss), '--exact_mapping', '1', '--which', 'LR','--norm_thresh','%s'%(norm_thresh),'--eig_thresh','%s'%(eig_thresh),'--dotproduct_thresh','%s'%(dotproduct_thresh),'--ordbasis','%s'%(order),'--nev', '%s'%(nev),'--nev_nodefault', '%s'%(nev_nodefault), '--postprocess', '%s'%(postprocess), '--tdplot', '%s'%(tdplot), '-option', '--tol_comp', '1d-4','--reclr_leaf','5','--lrlevel', '0', '--xyzsort', '2','--nmin_leaf', '100','--format', '1','--sample_para','2d0','--baca_batch','%s'%(baca_batch),'--knn','%s'%(knn),'--level_check','100','--verbosity', '%s'%(verbosity)], maxprocs=nproc,info=info)
		# comm = MPI.COMM_SELF.Spawn("%s/fdmom_port"%(RUNDIR), args=['-quant', '--model', '%s'%(model), '--freq', '%s'%(freq),'--si', '1', '--noport', '%s'%(noport), '--noloss', '%s'%(noloss), '--exact_mapping', '1', '--which', 'LM','--norm_thresh','%s'%(norm_thresh),'--ordbasis','%s'%(order),'--nev', '20', '--postprocess', '%s'%(postprocess), '-option', '--tol_comp', '1d-7','--reclr_leaf','5','--lrlevel', '0', '--xyzsort', '2','--nmin_leaf', '100','--format', '1','--sample_para','2d0','--baca_batch','%s'%(baca_batch),'--knn','%s'%(knn),'--level_check','100','--verbosity', '%s'%(verbosity)], maxprocs=nproc,info=info)

		""" gather the return value using the inter-communicator """							
		tmpdata = np.array([0, 0],dtype=np.float64)
		comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
		comm.Disconnect()	
		# read the file to see if a new valid sample has been generated
		(validfreq,retval) = read_validdata_onemode(model,nth,freq_int)



	with open(model+"_order_"+str(order)+"_SIMPLEX_freq_history_"+str(nth+1)+".csv", 'a') as f_object:
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
	global order
	global norm_thresh
	global eig_thresh
	global dotproduct_thresh
	global noport
	global postprocess
	global nev
	global nev_nodefault


	# Parse command line arguments

	args   = parse_args()

	# Extract arguments

	ntask = args.ntask
	nthreads = args.nthreads
	optimization = args.optimization
	nrun = args.nrun
	order = args.order
	postprocess = args.postprocess
	meshmodel = args.meshmodel

	# the default
	norm_thresh=1000
	eig_thresh=1e-6
	noport=0 # whether the port is treated as closed boundary or open port
	nev=40
	nev_nodefault=int(nev/2)
	if(noport==0):
		### with ports 
		dotproduct_thresh=0.9 #0.85
	else:
		### without ports: modes are typically very different, so dotproduct_thresh can be small 
		dotproduct_thresh=0.7  # you may want to use 0.9 for the final postprocessing 


####### cavity_rec_17K_feko
	if(meshmodel=="cavity_rec_17K_feko"):
		norm_thresh=1000
		eig_thresh=5e-7
		noport=1
		nev=40
		nev_nodefault=int(nev/2)
		if(noport==0):
			### with ports 
			dotproduct_thresh=0.9 #0.85
		else:
			### without ports: modes are typically very different, so dotproduct_thresh can be small 
			dotproduct_thresh=0.7

####### cavity_5cell_30K_feko
	if(meshmodel=="cavity_5cell_30K_feko"):
		norm_thresh=1000
		eig_thresh=1e-6
		noport=0
		nev=200
		nev_nodefault=int(nev/2)


####### cavity_5cell_30K_feko_copy
	if(meshmodel=="cavity_5cell_30K_feko_copy"):
		norm_thresh=1000
		eig_thresh=3e-7
		noport=0
		nev=200
		nev_nodefault=50
		if(noport==0):
			### with ports 
			dotproduct_thresh=0.75 #0.85
		else:
			### without ports: modes are typically very different, so dotproduct_thresh can be small 
			dotproduct_thresh=0.7
	TUNER_NAME = args.optimization	
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))


	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME
	

	


	# """ Building MLA with the given list of tasks """	
	# giventask = [["pillbox_4000"]]		
	# giventask = [["pillbox_1000"]]		
	# giventask = [["rfq_mirror_50K_feko"]]		
	giventask = [[meshmodel]]
	# giventask = [["cavity_rec_17K_feko"]]		
	# giventask = [["cavity_wakefield_4K_feko"]]		

	if(TUNER_NAME=='SIMPLEX'):
		if(compareVersion(sp.__version__,'1.7.0')==-1):
			raise Exception(f"Simplex require scipy version >=1.7.0 to support bounds")
		
		t3 = time.time_ns()
		(Pall,Oall)=read_validdata(giventask[0][0])
		Nmode = len(Pall)
		nth=0
		while nth<Nmode:
			P = Pall[nth]
			O = Oall[nth]
			print(nth+1, 'th mode:')
			print(P)
			print(O)

			os.system("rm -rf "+giventask[0][0]+"_order_"+str(order)+"_SIMPLEX_freq_history_"+str(nth+1)+".csv")

			X=np.array(P)	
			Xsort=np.sort(X,axis=0)

			if(X[np.argmin(O)]>Xsort[0] and X[np.argmin(O)]<Xsort[-1] and Xsort.size>10 and postprocess==0): # already enough samples and minimum is captured properly 
				print("Simplex refinement is not needed for mode ", nth+1)
			else:
				print('min ', np.amin(X),' max ', np.amax(X), 'mean ', np.mean(X), 'best ', X[np.argmin(O)])
				if(X[np.argmin(O)]>Xsort[0] and X[np.argmin(O)]<Xsort[-1]): # minimum will appear in between the left neighbour and the right neighour of the GPTune minimum
					idxmin=np.argmin(abs(Xsort-X[np.argmin(O)]))
					bounds_constraint = [(Xsort[idxmin-1],Xsort[idxmin+1])]
				elif(X[np.argmin(O)]==Xsort[0] and Xsort.size>1 and Xsort[0]<Xsort[1]): # minimum will appear to the left of Xsort[0] or in between the leftmost two data points	
					# bounds_constraint = [(Xsort[0] - (Xsort[-1]-Xsort[0])/2.0 ,Xsort[1])]
					bounds_constraint = [(Xsort[0]*0.996 ,Xsort[1])]
				elif(X[np.argmin(O)]==Xsort[-1] and Xsort.size>1 and Xsort[0]<Xsort[1]): # minimum will appear to the right of Xsort[-1] or in between the rightmost two data points						
					# bounds_constraint = [(Xsort[-2],Xsort[-1] + (Xsort[-1]-Xsort[0])/2.0)]
					bounds_constraint = [(Xsort[-2],Xsort[-1]*1.004)]
				else: # only 1 GPtune sample available, enlarge the search range
					bounds_constraint = [(Xsort[0]*0.996,Xsort[-1]*1.004)]

				#### either of the following two ways of initial sample is not perfect
				Pdefault = np.asarray(X[np.argmin(np.asarray(O))]) 
				# Pdefault = np.asarray((bounds_constraint[0][0]+bounds_constraint[0][1])/2.0)
				sol = sp.optimize.minimize(objectives, Pdefault, args=(nodes, cores, nthreads, giventask[0][0], nth), method='Nelder-Mead', options={'verbose': 1, 'maxfev': nrun, 'xatol': 1e-10, 'fatol': 1e-12}, bounds=bounds_constraint)    

				print('x      : ', sol.x)
				print('fun      : ', sol.fun)
				#print('hess_inv : ', sol.hess_inv)
				#print('jac      : ', jac)
				print('message  : ', sol.message)
				print('nfev     : ', sol.nfev)
				print('nit      : ', sol.nit)
				print('status   : ', sol.status)
				print('success  : ', sol.success)

				with open(giventask[0][0]+"_order_"+str(order)+"_SIMPLEX_freq_history_"+str(nth+1)+".csv", mode='r') as f_object:
					csv_reader = reader(f_object, delimiter=',')
					Ps=[]
					Os=[]
					for row in csv_reader:
						Ps.append(float(row[0]))
						Os.append(float(row[1]))
					Os=np.asarray(Os)	
					print("    Simplex refined mode ", nth+1)
					print("    bounds ", bounds_constraint)
					print("    Ps ", Ps)
					print("    Os ", Os.tolist())	
					print('    Popt ', sol.x, 'Oopt ', sol.fun, 'nth ', np.argmin(Os))	


			# reload the data and Nmode as they may have been updated by other modes
			(Pall,Oall)=read_validdata(giventask[0][0])
			# Nmode = len(Pall) # this line is commented to enforce that no new mode is allowed in SIMPLEX due to the heuristic problem with dotproduct_thresh
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
	parser.add_argument('-postprocess', type=int, default=0, help='whether postprocessing is performed')
	parser.add_argument('-meshmodel', type=str, default='cavity_5cell_30K_feko', help='Name of the mesh file')

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
