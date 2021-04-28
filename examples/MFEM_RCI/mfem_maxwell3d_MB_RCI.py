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
	ntask = args.ntask
	Nloop = args.Nloop
	bmin = args.bmin
	bmax = args.bmax
	eta = args.eta
	expid = args.expid
	TUNER_NAME = args.optimization
	nprocmin_pernode = args.nprocmin_pernode
	obj = args.obj
	target=obj
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	ot.RandomGenerator.SetSeed(args.seed)
	print(args)
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
	os.environ['TUNER_NAME'] = TUNER_NAME
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
	constants={"nodes":nodes,"cores":cores,"bmin":bmin,"bmax":bmax,"eta":eta}	

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
	options['sample_class'] = 'SampleOpenTURNS'
	# options['sample_class'] = 'SampleLHSMDU'
	# # options['sample_algo'] = 'LHS-MDU'	

	options.validate(computer = computer)
	
	options['budget_min'] = bmin
	options['budget_max'] = bmax
	options['budget_base'] = eta
	smax = int(np.floor(np.log(options['budget_max']/options['budget_min'])/np.log(options['budget_base'])))
	budgets = [options['budget_max'] /options['budget_base']**x for x in range(smax+1)]
	NSs = [int((smax+1)/(s+1))*options['budget_base']**s for s in range(smax+1)] 
	NSs_all = NSs.copy()
	budget_all = budgets.copy()
	for s in range(smax+1):
		for n in range(s):
			NSs_all.append(int(NSs[s]/options['budget_base']**(n+1)))
			budget_all.append(int(budgets[s]*options['budget_base']**(n+1)))
	Ntotal = int(sum(NSs_all) * Nloop)
	Btotal = int(np.dot(np.array(NSs_all), np.array(budget_all))/options['budget_max']*Nloop) # total number of evaluations at highest budget -- used for single-fidelity tuners
	print(f"bmin = {bmin}, bmax = {bmax}, eta = {eta}, smax = {smax}")
	print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)
	print("total number of samples: ", Ntotal)
	print("total number of evaluations at highest budget: ", Btotal)
	print(f"Sampler: {options['sample_class']}, {options['sample_algo']}")
	print()	

	data = Data(problem)
	giventask = [["inline-tet",16.0]]		
	Pdefault = [int(math.log2(nprocmin_pernode)),1,10,8,8]

	NI=len(giventask)
	assert NI == ntask # make sure number of tasks match


	np.set_printoptions(suppress=False, precision=4)
	if(TUNER_NAME=='GPTune'):
		NS = Btotal
		if args.nrun > 0:
			NS = args.nrun
		NS1 = max(NS//2, 1)
		
		data.I = giventask
		data.P = [[Pdefault]] * NI

		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		""" Building MLA with the given list of tasks """
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS1)
		# print("stats: ", stats)
		print("Sampler class: ", options['sample_class'], "Sample algo:", options['sample_algo'])
		print("Model class: ", options['model_class'])
		results_file = open(f"mfem_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
		results_file.write(f"Tuner: {TUNER_NAME}\n")
		results_file.write(f"stats: {stats}\n")        

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s freq:%f"%(data.I[tid][0],data.I[tid][1]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid])
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], f'Oopt  {min(data.O[tid])[0]:.3f}', 'nth ', np.argmin(data.O[tid]))
			results_file.write(f"tid: {tid:d}\n")
			results_file.write(f"    matrix:{data.I[tid][0]:s} freq:{data.I[tid][1]:f}\n")
			# results_file.write(f"    Ps {data.P[tid]}\n")
			results_file.write(f"    Os {data.O[tid].tolist()}\n")
			# results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
		results_file.close()            
		
	if(TUNER_NAME=='GPTuneBand'):
		NS = Nloop
		data = Data(problem)
		gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
		(data, stats, data_hist)=gt.MB_LCM(NS = Nloop, Igiven = giventask, Pdefault=Pdefault)
		print("Tuner: ", TUNER_NAME)
		print("Sampler class: ", options['sample_class'])
		print("Model class: ", options['model_class'])
		# print("stats: ", stats)
		results_file = open(f"mfem_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
		results_file.write(f"Tuner: {TUNER_NAME}\n")
		results_file.write(f"stats: {stats}\n")        
		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d" % (tid))
			print("    matrix:%s freq:%f"%(data.I[tid][0],data.I[tid][1]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			nth = np.argmin(data.O[tid])
			Popt = data.P[tid][nth]
			# find which arm and which sample the optimal param is from
			for arm in range(len(data_hist.P)):
				try:
					idx = (data_hist.P[arm]).index(Popt)
					arm_opt = arm
				except ValueError:
					pass
			print('    Popt ', Popt, 'Oopt ', min(data.O[tid])[0], 'nth ', nth, 'nth-bandit (s, nth) = ', (arm_opt, idx))
			results_file.write(f"tid: {tid:d}\n")
			results_file.write(f"    matrix:{data.I[tid][0]:s} freq:{data.I[tid][1]:f}\n")
			# results_file.write(f"    Ps {data.P[tid]}\n")
			results_file.write(f"    Os {data.O[tid].tolist()}\n")
			# results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
		results_file.close()


def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	parser.add_argument('-obj', type=str, default='time', help='Tuning objective (time or memory)')	
	# Machine related arguments
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	# Algorithm related arguments
	parser.add_argument('-bmin', type=int, default=1, help='budget min')   
	parser.add_argument('-bmax', type=int, default=1, help='budget max')   
	parser.add_argument('-eta', type=int, default=2, help='eta')
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
	parser.add_argument('-nrun', type=int, default=-1, help='Number of runs per task')
	parser.add_argument('-Nloop', type=int, default=1, help='Number of outer loops in multi-armed bandit per task')
	# Experiment related arguments
	parser.add_argument('-seed', type=int, default=1, help='random seed')
	parser.add_argument('-expid', type=str, default='-', help='run id for experiment')

	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()
