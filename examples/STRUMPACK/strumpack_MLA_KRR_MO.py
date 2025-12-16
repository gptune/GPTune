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
mpirun -n 1 python ./strumpack_MLA_KRR.py -ntask 1 -nrun 20 -npernode 32


where:
    -npernode is the number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
"""
 
################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

import mpi4py
from mpi4py import MPI
from array import array
import math



from GPTune.gptune import * # import all

from autotune.problem import *
from autotune.space import *
from autotune.search import *

from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter
import math

import time

################################################################################
def objectives(point):
    #########################################
    ##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']
    npernode = point['npernode']
    #########################################

    datafile = "data/"+str(point['datafile'])
    h = 10**point['h']
    Lambda = 10**point['Lambda']
    p = point['p']

    nproc = nodes*npernode
    nthreads = int(cores / npernode)

    params = ['datafile', datafile, 'h', h, 'Lambda', Lambda, 'p', p, 'nthreads', nthreads, 'npernode', npernode, 'nproc', nproc]
    RUNDIR = os.path.abspath(__file__ + "/../STRUMPACK/examples")
    INPUTDIR = os.path.abspath(__file__ + "/../STRUMPACK/examples")
    TUNER_NAME = os.environ['TUNER_NAME']

    """ pass some parameters through environment variables """
    info = MPI.Info.Create()
    envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)
    print('OMP_NUM_THREADS=%d\n' %(nthreads))
    info.Set('env',envstr)
    info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works

    degree = 1	# only used when kernel=ANOVA (degree<=d) in KernelRegressionMPI.py

    """ use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
    # print('exec', "%s/testPoisson3dMPIDist"%(RUNDIR), 'args', ['%s'%(gridsize), '1', '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_compression', '%s'%(sp_compression1),'--sp_nd_param', '%s'%(sp_nd_param),'--sp_compression_min_sep_size', '%s'%(sp_compression_min_sep_size),'--sp_compression_min_front_size', '%s'%(sp_compression_min_front_size),'--sp_compression_leaf_size', '%s'%(sp_compression_leaf_size)]+extra_str, 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
    comm = MPI.COMM_SELF.Spawn("%s/KernelRegressionMPI.py"%(RUNDIR), args=['%s/%s'%(INPUTDIR,datafile), '%s'%(h),'%s'%(Lambda),'%s'%(degree), '%s'%(p)], maxprocs=nproc,info=info)

    """ gather the return value using the inter-communicator """
    tmpdata = np.array([0],dtype=np.float64)
    comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT)
    tmpdata1 = np.array([0],dtype=np.float64)
    comm.Reduce(sendbuf=None, recvbuf=[tmpdata1,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT)
    comm.Disconnect()

    print(params, ' krr prediction error: ', tmpdata[0])
    print(params, ' krr training time: ', tmpdata1[0])

    return [tmpdata[0], tmpdata1[0]]

def main():

    # Parse command line arguments

    args   = parse_args()

    # Extract arguments

    ntask = args.ntask
    npernode = args.npernode
    optimization = args.optimization
    nrun = args.nrun

    print ("NPERNODE: ", npernode)

    dataset = args.dataset

    TUNER_NAME = args.optimization
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    #datafiles = ["data/susy_10Kn"]
    datafiles = [dataset]

    # Task input parameters
    datafile = Categoricalnorm (datafiles, transform="onehot", name="datafile")

    # Tuning parameters
    h = Real(-10, 10, transform="normalize", name="h")
    Lambda = Real(-10, 10, transform="normalize", name="Lambda")
    p = Real(0.1, 1.0, transform="normalize", name="p") # oversampling parameter

    # npernode = Integer(int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")

    error = Real(0, float("Inf"), name="error")
    training_time = Real(0, float("Inf"), name="training_time")

    IS = Space([datafile])
    PS = Space([h,Lambda,p])
    OS = Space([error,training_time])
    constraints = {}
    models = {}
    constants = {"nodes":nodes,"cores":cores,"npernode":npernode}

    """ Print all input and parameter samples """
    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, constants=constants)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['model_class'] = 'Model_LCM' # 'Model_GPy_LCM'
    options['verbose'] = False

    # MO
    options['search_algo'] = 'nsga2' #'maco' #'moead' #'nsga2' #'nspso'
    options['search_pop_size'] = 1000
    options['search_gen'] = 50
    options['search_more_samples'] = 4

    options.validate(computer = computer)

    # """ Building MLA with the given list of tasks """
    giventask = [[dataset]]
    data = Data(problem)

    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))

        NI = len(giventask)
        NS = nrun
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            #print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        """ Print all input and parameter samples """
        import pymoo
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    problem:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            front = NonDominatedSorting(method="fast_non_dominated_sort").do(data.O[tid], only_non_dominated_front=True)
            # print('front id: ',front)
            fopts = data.O[tid][front]
            xopts = [data.P[tid][i] for i in front]
            print('    Popts ', xopts)
            print('    Oopts ', fopts.tolist())  

    if(TUNER_NAME=='opentuner'):
        NI = ntask
        NS = nrun
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = ntask
        NS = nrun
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='Random'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))

        NI = len(giventask)
        NS = nrun
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=NS)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            #print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        """ Print all input and parameter samples """
        import pymoo
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    problem:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            front = NonDominatedSorting(method="fast_non_dominated_sort").do(data.O[tid], only_non_dominated_front=True)
            # print('front id: ',front)
            fopts = data.O[tid][front]
            xopts = [data.P[tid][i] for i in front]
            print('    Popts ', xopts)
            print('    Oopts ', fopts.tolist())  
	
def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments

    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-npernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')

    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')

    # Input dataset
    parser.add_argument('-dataset', type=str, default="susy_10Kn", help='Name of the dataset')

    args   = parser.parse_args()
    return args

if __name__ == "__main__":

    main()
