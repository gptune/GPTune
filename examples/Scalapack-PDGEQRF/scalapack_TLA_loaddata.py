#! /usr/bin/env python3

"""
Example of invocation of this script:

mpirun -n 1 python scalapack_TLA_loaddata.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 5 -nrun 10 -machine cori -jobid 0

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
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))

from pdqrdriver import pdqrdriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from options import Options
from computer import Computer
import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
import math



################################################################################


''' The objective function required by GPTune. '''
def objectives(point):                  # should always use this name for user-defined objective function
    m = point['m']
    n = point['n']
    mb = point['mb']*bunit
    nb = point['nb']*bunit
    p = point['p']
    npernode = 2**point['npernode']
    nproc = nodes*npernode
    nthreads = int(cores / npernode)  

    # this becomes useful when the parameters returned by TLA1 do not respect the constraints
    if(nproc == 0 or p == 0 or nproc < p):
        print('Warning: wrong parameters for objective function!!!')
        return 1e12
    q = int(nproc / p)
    nproc = p*q
    params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]

    print(params, ' scalapack starts ') 
    elapsedtime = pdqrdriver(params, niter=3, JOBID=JOBID)
    print(params, ' scalapack time: ', elapsedtime)
    return elapsedtime
def cst1(mb,p,m):
    return mb*bunit * p <= m
def cst2(nb,npernode,n,p):
    return nb * bunit * nodes * 2**npernode <= n * p
def cst3(npernode,p):
    return nodes * 2**npernode >= p	

	
def main():

	global ROOTDIR
	global nodes
	global cores
	global bunit
	global JOBID
	global nprocmax
	global nprocmin

	# Parse command line arguments
	args = parse_args()

	mmax = args.mmax
	nmax = args.nmax
	ntask = args.ntask
	nodes = args.nodes
	cores = args.cores
	nprocmin_pernode = args.nprocmin_pernode
	machine = args.machine
	# optimization = args.optimization
	nruns = args.nruns
	JOBID = args.jobid
	os.environ['MACHINE_NAME']=machine
	os.environ['TUNER_NAME']='GPTune'
	os.system("mkdir -p scalapack-driver/bin/%s; cp ../../build/pdqrdriver scalapack-driver/bin/%s/.;"%(machine, machine))
	nprocmax = nodes*cores
	bunit=8

	
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
	# Experiment related arguments
	parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)

	args   = parser.parse_args()

	return args

if __name__ == "__main__":
	main()
