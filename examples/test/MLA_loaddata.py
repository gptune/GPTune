#! /usr/bin/env python3

"""
Example of invocation of this script:

python MLA_loaddata.py -nodes 1 -cores 4  -nrun 60 2>&1 | tee -a MLA_log.txt

"""

################################################################################

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import sys
import os
import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
import time

sys.path.insert(0, os.path.abspath(__file__ + "../../../GPTune/"))


import re
import subprocess

################################################################################

''' The objective function required by GPTune. '''


def objectives(point):
    ka = point['kappa']
    threshold = point['threshold']
    # wt = point['wt']
    # print(ka,threshold)

    cmd = '''python Execute.py  --batch_size 1  --keys workdir  --keys AMG_strong_threshold  --vals {} --vals {} '''.format(ka,threshold)
    printout = subprocess.check_output(cmd,shell=True)
    printout = printout.decode('utf-8')
    contents = printout.split('\n')
    # print(printout)
    retval = 1000.0
    for item in contents:
        if re.search('iteration_num',item):
            tmp = item.split('=')[-1]
            retval = eval(tmp)
    # retval = retval*(1+np.random.uniform()*0.5)

    print('finalcheck kappa = {}, threshold= {}, retval = {}'.format(ka,threshold,retval))
    return retval 

def main():

    global ROOTDIR
    global nodes
    global cores
    global target
    global dim 

    # Parse command line arguments
    args = parse_args()

    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    optimization = args.optimization
    nruns = args.nruns
    truns = args.truns
    dim   = args.dim

    # os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = 'GPTune'

    # Task parameters
    kappa = Categoricalnorm (['6','7','8','9','10','11','12','13','14','15','16'], transform="onehot", name="kappa")

    # Input parameters
    threshold   = Real        (0.1 , 0.9, transform="normalize", name="threshold")

    # Output parameters
    iteration   = Real        (float("-Inf") , float("Inf"), name="iteration")

    IS = Space([kappa])
    PS = Space([threshold])
    OS = Space([iteration])
    constraints = {}
    models = {}
    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_latent'] = 5
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = True
    # options['mpi_comm'] = None
    # options['model_class '] = 'Model_LCM'
    options['model_class '] = 'Model_LCM'
    options['verbose'] = True
    options['model_max_jitter_try'] = 1
    options.validate(computer=computer)


    """ Intialize the tuner with existing data stored as last check point"""

    data = Data(problem)

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):

        gt = GPTune(problem, computer=computer, data=data, options=options)

        """ Building MLA with NI random tasks """
        #NI = ntask
        #NS = nruns
        #(data, model, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=max(NS//2, 1))
        

        giventask = [['6'],['7'],['8'],['9'],['10'],['11'],['12'],['13'],['14'],['15']]
        NI = len(giventask)
        NS = nruns
        (data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = max(NS//2,1))
        print("stats: ", stats)

        part_name = 'iter_60.pkl'

        """ Dump the data to file as a new check point """
        pickle.dump(data, open('Data_nodes_'+part_name, 'wb'))

        """ Dump the tuner to file for TLA use """
        pickle.dump(gt, open('MLA_nodes_'+part_name , 'wb'))

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            #print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    m:{} ".format(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='opentuner'):
        NI = ntask
        NS = nruns
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))



def parse_args():

    parser = argparse.ArgumentParser()

    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    parser.add_argument('-dim', type=int, default=2, help='the dimension of problem')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
