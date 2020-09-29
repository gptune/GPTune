#! /usr/bin/env python3

"""
Example of invocation of this script:

python scalapack.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -nodes is the number of compute nodes
    -cores is the number of cores per node
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -machine is the name of the machine
    -jobid is optional. You can always set it to 0.
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))

from computer import Computer
from options import Options
from data import Data
from gptune import GPTune

from autotune.problem import *
from autotune.space import *
from autotune.search import *


sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))
from pdqrdriver import pdqrdriver



################################################################################


''' The objective function required by GPTune. '''
def objectives(point):                  # should always use this name for user-defined objective function
	m = point['m']
	n = point['n']
	mb = point['mb']
	nb = point['nb']
	nproc = point['nproc']
	p = point['p']
	if(nproc==0 or p==0 or nproc<p): # this becomes useful when the parameters returned by TLA1 do not respect the constraints
		print('Warning: wrong parameters for objective function!!!')
		return 1e12

	npernode =  math.ceil(float(nproc)/nodes)  	
	nthreads = int(cores / npernode)
	q = int(nproc / p)
	params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]

	elapsedtime = pdqrdriver(params, niter = 3, JOBID=JOBID)
	print(params, ' scalapack time: ', elapsedtime)
	return elapsedtime 

	
def main():

	global ROOTDIR
	global nodes
	global cores
	global JOBID
	global nprocmax
	global nprocmin

	# Parse command line arguments
	args   = parse_args()

	mmax = args.mmax
	nmax = args.nmax
	ntask = args.ntask
	nodes = args.nodes
	cores = args.cores
	nprocmin_pernode = args.nprocmin_pernode
	machine = args.machine
	# optimization = args.optimization
	nruns = args.nruns
	truns = args.truns
	JOBID = args.jobid
	os.environ['MACHINE_NAME']=machine
	os.environ['TUNER_NAME']='GPTune'
	os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;"%(machine, machine))
	nprocmax = nodes*cores-1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
	nprocmin = min(nodes*nprocmin_pernode,nprocmax-1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space 

	
	""" Load the tuner and data from file """
	gt = pickle.load(open('MLA_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl'%(nodes,cores,mmax,nmax,machine,JOBID), 'rb'))
	 
	""" Call TLA for 2 new tasks using the loaded data and LCM model"""		 
	newtask = [[400,500],[800,600]]
	(aprxopts,objval,stats) = gt.TLA1(newtask, NS=None)
	print("stats: ",stats)
		
	""" Print the optimal parameters and function evaluations"""		
	for tid in range(len(newtask)):
		print("new task: %s"%(newtask[tid]))
		print('    predicted Popt: ', aprxopts[tid], ' objval: ',objval[tid]) 	
		
		
def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
	parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
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
