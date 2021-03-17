#! /usr/bin/env python3

"""
Example of invocation of this script:

mpirun -n 1 python scalapack_MLA_TLA.py -mmax 5000 -nmax 5000 -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 5 -nrun 10 -machine cori -jobid 0

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
# should always use this name for user-defined objective function
def objectives(point):
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



# should always use this name for user-defined model function
def models(point):

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

    b = max(mb,nb)
    # elapsedtime = (2*n**2*(3*m-n)/3/nproc + b*n**2/2/q + 3*b*n*(2*m-n)/2/p + b**2*n/3/p)/(max(m,n)*min(m,n)**2)
    elapsedtime = max(m,n)*min(m,n)**2/nproc/(max(m,n)*min(m,n)**2)

    return [elapsedtime]


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
    nruns = args.nruns
    truns = args.truns
    JOBID = args.jobid
    TUNER_NAME = args.optimization

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    os.system("mkdir -p scalapack-driver/bin/%s; cp ../../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores

    bunit=8     # the block size is multiple of bunit
    mmin=128
    nmin=128

    m = Integer(mmin, mmax, transform="normalize", name="m")
    n = Integer(nmin, nmax, transform="normalize", name="n")
    mb = Integer(1, 16, transform="normalize", name="mb")
    nb = Integer(1, 16, transform="normalize", name="nb")
    npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
    p = Integer(1, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, npernode, p])
    OS = Space([r])

    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    print(IS, PS, OS, constraints)

    # problem = TuningProblem(IS, PS, OS, objectives, constraints, models) # use performance models
    problem = TuningProblem(IS, PS, OS, objectives, constraints, None) # no performance model

    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1

    # options['objective_evaluation_parallelism'] = True

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class'] = 'Model_GPy_LCM'
    options['verbose'] = False
    options.validate(computer=computer)
    # giventask = [[2000, 2000]]
    # giventask = [[randint(mmin,mmax),randint(nmin,nmax)] for i in range(ntask)]
    giventask = [[460, 500], [800, 690]]


    data = Data(problem)



    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nruns
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)
        pickle.dump(gt, open('MLA_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl' % (
            nodes, cores, mmax, nmax, machine, JOBID), 'wb'))

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Yopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        """ Call TLA for 2 new tasks using the constructed LCM model"""
        newtask = [[400, 500], [800, 600]]
        (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
        print("stats: ", stats)

        """ Print the optimal parameters and function evaluations"""
        for tid in range(len(newtask)):
            print("new task: %s" % (newtask[tid]))
            print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])


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
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = ntask
        NS = nruns
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    parser.add_argument('-nmax', type=int, default=-
                        1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,
                        help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,
                        help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')                    
    parser.add_argument('-machine', type=str,
                        help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str, default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-
                        1, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
