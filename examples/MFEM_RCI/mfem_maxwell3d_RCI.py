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
python mfem_maxwell3d_RCI.py -nprocmin_pernode 1 -nrun 800 -obj time

where:
	-nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -nrun is the number of calls per task 
	-obj is the tuning objective: "time" or "memory"	
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

import mpi4py
from array import array
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all


from autotune.problem import *
from autotune.space import *
from autotune.search import *

# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                          
	print('objective is not needed when options["RCI_mode"]=True')
	

def main():


	# Parse command line arguments

	args   = parse_args()

	# Extract arguments

	nprocmin_pernode = args.nprocmin_pernode
	optimization = args.optimization
	nrun = args.nrun
	obj = args.obj
	target=obj
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
	TUNER_NAME = 'GPTune'
	os.environ['MACHINE_NAME'] = machine

	# Task parameters
	meshes = ["escher", "fichera", "periodic-cube", "amr-hex", "inline-tet"]
	mesh    = Categoricalnorm (meshes, transform="onehot", name="mesh")    
	omega = Real(16.0, 64.0, transform="normalize", name="omega")

	# Tuning parameters
	# sp_reordering_method   = Categoricalnorm (['metis','parmetis','scotch'], transform="onehot", name="sp_reordering_method")
	npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")	
	sp_blr_min_sep_size     = Integer     (1, 10, transform="normalize", name="sp_blr_min_sep_size")
	sp_hodlr_min_sep_size     = Integer     (10, 40, transform="normalize", name="sp_hodlr_min_sep_size")
	hodlr_leaf_size     = Integer     (5, 9, transform="normalize", name="hodlr_leaf_size")
	blr_leaf_size     = Integer     (5, 9, transform="normalize", name="blr_leaf_size")
	if(target=='time'):			
		result   = Real        (float("-Inf") , float("Inf"),name="time")
	if(target=='memory'):	
		result   = Real        (float("-Inf") , float("Inf"),name="memory")

	IS = Space([mesh,omega])
	PS = Space([npernode, sp_blr_min_sep_size,sp_hodlr_min_sep_size,blr_leaf_size,hodlr_leaf_size])
	OS = Space([result])
	constraints = {}
	models = {}
	constants={"nodes":nodes,"cores":cores}	

	# """ Print all input and parameter samples """	
	# print(IS, PS, OS, constraints, models)


	problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
	computer = Computer(nodes = nodes, cores = cores, hosts = None)  

	""" Set and validate options """	
	options = Options()
	options['RCI_mode'] = True
	options['model_processes'] = 1
	# options['model_threads'] = 1
	options['model_restarts'] = 1
	# options['search_multitask_processes'] = 1
	# options['model_restart_processes'] = 1
	options['distributed_memory_parallelism'] = False
	options['shared_memory_parallelism'] = False
	options['model_class'] = 'Model_GPy_LCM' # 'Model_LCM'
	options['verbose'] = False

	options.validate(computer = computer)
	
	
	# """ Building MLA with the given list of tasks """
	giventask = [["inline-tet",32.0]]		
	data = Data(problem)
	
	data.I = giventask
	Pdefault = [int(math.log2(nprocmin_pernode)),1,10,8,8]
	data.P = [[Pdefault]]

	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = nrun
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
		# print("stats: ", stats)

		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	parser.add_argument('-obj', type=str, default='time', help='Tuning objective (time or memory)')	
	# Machine related arguments
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')
	# Experiment related arguments

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
