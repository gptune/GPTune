#! /usr/bin/env python3

"""
Example of invocation of this script:

python TLA_loaddata.py -nodes 1 -cores 4  2>&1 | tee -a TLA_log.txt

"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

sys.path.insert(0, os.path.abspath(__file__ + "../../../GPTune/"))

from computer import Computer
from options import Options
from data import Data
from gptune import GPTune

from autotune.problem import *
from autotune.space import *
from autotune.search import *


import re
import subprocess



################################################################################


''' The objective function required by GPTune. '''
def objectives(point):                  # should always use this name for user-defined objective function
    ka = point['kappa']
    threshold = point['threshold']

    cmd = '''python3 Execute.py  --batch_size 1  --keys workdir  --keys AMG_strong_threshold  --vals {} --vals {} '''.format(ka,threshold)
    printout = subprocess.check_output(cmd,shell=True)
    printout = printout.decode('utf-8')
    contents = printout.split('\n')
    # print(printout)
    retval = 1000.0
    for item in contents:
        if re.search('iteration_num',item):
            tmp = item.split('=')[-1]
            retval = eval(tmp)

    print('finalcheck retval = {}'.format(retval))
    return retval 

	
def main():

    global ROOTDIR
    global nodes
    global cores
    global target
    global dim 

    # Parse command line arguments
    args   = parse_args()

    # Extract arguments
    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    optimization = args.optimization
    nruns = args.nruns
    truns = args.truns
    dim   = args.dim
    os.environ['MACHINE_NAME']=machine
    os.environ['TUNER_NAME']='GPTune'

    """ Load the tuner and data from file """
    part_name = 'iter_60.pkl'
    gt = pickle.load(open('MLA_nodes_'+part_name, 'rb'))
	 
    """ Call TLA for  new tasks using the loaded data and LCM model"""		 
    newtask = [['16']]

    (aprxopts,objval,stats) = gt.TLA1(newtask, NS=None)
    print("stats: ",stats)
		
    """ Print the optimal parameters and function evaluations"""		
    for tid in range(len(newtask)):
        print("new task: %s"%(newtask[tid]))
        print('    predicted Popt: ', aprxopts[tid], ' objval: ',objval[tid]) 	
		
		
def parse_args():
    
    parser = argparse.ArgumentParser()

    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    parser.add_argument('-dim', type=int, default=2, help='the dimension of problem')
    # Experiment related arguments
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job') #0 means interactive execution (not batch)
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')

    args   = parser.parse_args()

    return args

if __name__ == "__main__":
	main()
