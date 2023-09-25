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
from array import array
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

import math



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

# mergemode('cavity_rec_17K_feko')

# checking and merging when there are any pair of modes that should be treated as one mode
def mergemode(model):
	file =open(model+'_order_%s_Nmodes.txt'%(order),'r')
	Lines = file.readlines()
	Nmode = int(Lines[0].strip())
	Nmode_new = Nmode
	file.close()
	subnames=['left','right','min']
	eigvecss=[]
	eiginfo=[]
	for mm in range(Nmode): 
		# preload all the eigvectors
		eigvecs=[]
		for i in range(len(subnames)):
			filename_i=model+'_order_%s_mode_vec_%s'%(order,subnames[i])+str(mm+1)+'.out'	
			if(os.path.exists(filename_i)):
				with open(filename_i) as f:
					polyShape = []
					for line in f:
						line = line.split() # to deal with blank 
						if line:            # lines (ie skip them)
							line = float(line[0])+float(line[1])*1j
							polyShape.append(line)
					eigvec_i = np.array(polyShape)
					eigvecs.append(eigvec_i)
		eigvecss.append(eigvecs)
		# read the min/max frequency and minimum eigenvalue of each mode
		filename_v=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
		if(os.path.exists(filename_v)):
			file =open(filename_v,'r')
			Lines = file.readlines()
			freqs = []
			eigvals = []
			for nn in range(len(Lines)): 
				freq =int(round(float(Lines[nn].split()[0])/1e5)) 
				eigval =float(Lines[nn].split()[1])
				freqs.append(freq)
				eigvals.append(eigval)
			file.close()
			freqs = np.array(freqs)
			eigvals = np.array(eigvals)
			info=np.array([np.amin(freqs),np.amax(freqs),np.amin(eigvals)])
			eiginfo.append(info)	
	# pair-wise comparision starts here	
	for mm in range(Nmode):
		for nn in range(Nmode):
			if(len(eigvecss[mm])>0): 
				if(len(eigvecss[nn])>0): 
					if(nn>mm):
						similar=0
						for i in range(len(subnames)):
							for j in range(len(subnames)):
								eigvec_i = eigvecss[mm][i]
								eigvec_j = eigvecss[nn][j]
								if(eigvec_i.shape[0]==eigvec_j.shape[0]):
									dot = np.abs(np.vdot(eigvec_i,eigvec_j))
									if(dot>0.7):
										print('checking mode %s %s vs mode %s %s %f'%(mm+1,subnames[i],nn+1,subnames[j],dot))
									# if(dot>dotproduct_thresh and not (mm==5 and nn==7)): # need to rule out these two modes as their inner product are large, but they seem different modes
									if(dot>dotproduct_thresh):
										similar=1
						if(similar==1):
							print('mode %s is merged into mode %s'%(nn+1,mm+1))
							if(eiginfo[nn][0]<eiginfo[mm][0]): # new left frequency
								# print('new left frequency')
								eiginfo[mm][0] = eiginfo[nn][0]
								eigvecss[mm][0] = copy.deepcopy(eigvecss[nn][0])
								filename_i=model+'_order_%s_mode_vec_%s'%(order,subnames[0])+str(mm+1)+'.out'	
								filename_j=model+'_order_%s_mode_vec_%s'%(order,subnames[0])+str(nn+1)+'.out'	
								os.system("cp %s %s "%(filename_j,filename_i)) 
							if(eiginfo[nn][1]>eiginfo[mm][1]): # new right frequency
								# print('new right frequency')
								eiginfo[mm][1] = eiginfo[nn][1]
								eigvecss[mm][1] = copy.deepcopy(eigvecss[nn][1])
								filename_i=model+'_order_%s_mode_vec_%s'%(order,subnames[1])+str(mm+1)+'.out'	
								filename_j=model+'_order_%s_mode_vec_%s'%(order,subnames[1])+str(nn+1)+'.out'	
								os.system("cp %s %s "%(filename_j,filename_i)) 								
							if(eiginfo[nn][2]<eiginfo[mm][2]): # new min eig value
								# print('new min eig value')
								eiginfo[mm][2] = eiginfo[nn][2]
								eigvecss[mm][2] = copy.deepcopy(eigvecss[nn][2])
								filename_i=model+'_order_%s_mode_vec_%s'%(order,subnames[2])+str(mm+1)+'.out'	
								filename_j=model+'_order_%s_mode_vec_%s'%(order,subnames[2])+str(nn+1)+'.out'	
								os.system("cp %s %s "%(filename_j,filename_i))
							filename_v_m=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
							filename_v_n=model+'_order_%s_EigVals_'%(order)+str(nn+1)+'.out'
							os.system("cat %s >> %s "%(filename_v_n,filename_v_m)) # merge the objective function samples
							eiginfo[nn]=[]
							eigvecss[nn]=[]
							os.system("rm %s"%(filename_v_n))
							for j in range(len(subnames)):
								filename_j=model+'_order_%s_mode_vec_%s'%(order,subnames[j])+str(nn+1)+'.out'	
								os.system("rm %s"%(filename_j))
							# name = input("Continue? ")
	# rename the files
	Nmode_new=0
	for mm in range(Nmode):
		filename_old=model+'_order_%s_EigVals_'%(order)+str(mm+1)+'.out'
		if(os.path.exists(filename_old)):
			Nmode_new = Nmode_new + 1
			if(Nmode_new != mm+1):
				filename_new=model+'_order_%s_EigVals_'%(order)+str(Nmode_new)+'.out'
				os.system("mv %s %s "%(filename_old,filename_new))
				for i in range(len(subnames)):
					filename_i_old=model+'_order_%s_mode_vec_%s'%(order,subnames[i])+str(mm+1)+'.out'	
					filename_i_new=model+'_order_%s_mode_vec_%s'%(order,subnames[i])+str(Nmode_new)+'.out'	
					os.system("mv %s %s "%(filename_i_old,filename_i_new))
	filename=model+'_order_%s_Nmodes.txt'%(order)
	os.system("echo %s > %s "%(Nmode_new, filename)) 

def main():
	global order
	global norm_thresh
	global eig_thresh
	global dotproduct_thresh
	global noport
    
	noport=0
	
	if(noport==0):
		### with ports 
		dotproduct_thresh=0.7 #0.9
	else:
		### without ports: modes are typically very different, so dotproduct_thresh can be small 
		dotproduct_thresh=0.7

	# Parse command line arguments

	args   = parse_args()

	# Extract arguments

	ntask = args.ntask
	nthreads = args.nthreads
	optimization = args.optimization
	nrun = args.nrun
	order = args.order


	# """ Building MLA with the given list of tasks """	
	# giventask = [["pillbox_4000"]]		
	# giventask = [["pillbox_1000"]]		
	# giventask = [["rfq_mirror_50K_feko"]]		
	# giventask = [["cavity_5cell_30K_feko"]]		
	giventask = [["cavity_5cell_30K_feko_copy"]]		
	# giventask = [["cavity_rec_5K_feko"]]
	# giventask = [["cavity_rec_17K_feko"]]
	# giventask = [["cavity_rec_17K_2nd_mesh"]]
	# # giventask = [["rect_waveguide_2000"]]		
	# giventask = [["rect_waveguide_30000"]]		
	# giventask = [["cavity_wakefield_4K_feko"]]
	mergemode(giventask[0][0])	

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
