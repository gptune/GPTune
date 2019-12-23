#! /usr/bin/env python3

"""
Example of invocation of this script:

python scalapack.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -nodes is the number of compute nodes
    -cores is the number of cores per node
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
def objective(point):                  
	m = point['m']
	n = point['n']
	mb = point['mb']
	nb = point['nb']
	nproc = point['nproc']
	p = point['p']

	if(nproc==0 or p==0 or nproc<p): # this become useful when the parameters returned by TLA1 do not respect the constraints
		print('Warning: wrong parameters for objective function!!!')
		return 1e12
	nth   = int(nprocmax / nproc) 
	q     = int(nproc / p)
	params = [('QR', m, n, nodes, cores, mb, nb, nth, nproc, p, q, 1.)]
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

	parser = argparse.ArgumentParser()

	# Problem related arguments
	parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
	parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	# parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
	parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
	parser.add_argument('-nruns', type=int, help='Number of runs per task')
	parser.add_argument('-truns', type=int, help='Time of runs')
	# Experiment related arguments
	parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)

	args   = parser.parse_args()

	
	
	''' Extract the elements. '''	
	mmax = args.mmax # maximum row dimension
	nmax = args.nmax  # maximum column dimension
	ntask = args.ntask  # number of tasks used for MLA
	nodes = args.nodes  # number of nodes used for the tuner
	cores = args.cores  # number of threads used for the tuner
	machine = args.machine  # machine is part of the input/output file names
	nruns = args.nruns   # number of samples per task in MLA 
	JOBID = args.jobid   # JOBID is part of the input/output file names
	nprocmax = nodes*cores-1  # The maximum MPI counts for the application code. Note that 1 process is reserved as the spawning process
	nprocmin = nodes	
	os.environ['MACHINE_NAME']=machine
	os.environ['TUNER_NAME']='GPTune'

	os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;"%(machine, machine))

	""" Define and print the spaces and constraints """		
	# Inputs 
	m     = Integer (128 , mmax, transform="normalize", name="m")
	n     = Integer (128 , nmax, transform="normalize", name="n")
	IS = Space([m, n])	
	# Parameters	
	mb    = Integer (1 , 128, transform="normalize", name="mb")
	nb    = Integer (1 , 128, transform="normalize", name="nb")
	nproc = Integer (nprocmin, nprocmax, transform="normalize", name="nproc") 
	p     = Integer (1 , nprocmax, transform="normalize", name="p")
	PS = Space([mb, nb, nproc, p])	
	# Output
	r     = Real    (float("-Inf") , float("Inf"), name="r")
	OS = Space([r])
	# Constraints	
	cst1 = "mb * p <= m"
	cst2 = "nb * nproc <= n * p"
	cst3 = "nproc >= p" 
	constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3}
	print(IS, PS, OS, constraints)

	problem = TuningProblem(IS, PS, OS, objective, constraints, None)
	computer = Computer(nodes = nodes, cores = cores, hosts = None)  

	""" Set and validate options """	
	options = Options()
	# options['model_processes'] = 1
	# options['model_threads'] = 1
	options['model_restarts'] = 1
	# options['search_multitask_processes'] = 1
	# options['model_restart_processes'] = 1
	# options['model_restart_threads'] = 1
	options['distributed_memory_parallelism'] = False
	options['shared_memory_parallelism'] = False
	# options['mpi_comm'] = None
	options['model_class '] = 'Model_LCM'
	options['verbose'] = False
	options.validate(computer = computer)

	""" Intialize the tuner with existing data stored as last check point"""	
	try:
		data = pickle.load(open('Data_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl'%(nodes,cores,mmax,nmax,machine,JOBID), 'rb'))
	except (OSError, IOError) as e:
		data = Data(problem)
	gt = GPTune(problem, computer = computer, data = data, options = options)

	""" Building MLA with NI random tasks """
	NI = ntask
	NS = nruns
	(data, model,stats) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))
	print("stats: ",stats)

	""" Dump the data to file as a new check point """
	pickle.dump(data, open('Data_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl'%(nodes,cores,mmax,nmax,machine,JOBID), 'wb'))

	""" Dump the tuner to file for TLA use """
	pickle.dump(gt, open('MLA_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl'%(nodes,cores,mmax,nmax,machine,JOBID), 'wb'))	
	

	for tid in range(NI):
		print("tid: %d"%(tid))
		print("    m:%d n:%d"%(data.I[tid][0], data.I[tid][1]))
		print("    Ps ", data.P[tid])
		print("    Os ", data.O[tid])
		print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Yopt ', min(data.O[tid])[0])


		
def parse_args():

    # Parse command line arguments

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

    # Extract arguments

    return (args.mmax, args.nmax, args.ntask, args.nodes, args.cores, args.machine, args.optimization, args.nruns, args.truns, args.jobid, args.stepid, args.phase)

if __name__ == "__main__":
   main()
 
