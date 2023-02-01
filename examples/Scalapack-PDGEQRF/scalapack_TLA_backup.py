#! /usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))

from pdqrdriver import pdqrdriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import time
import math

''' The objective function required by GPTune. '''
# should alwas use this name for user-defined objective function
def objectives(point):

    if point['machine'] == 'summit':
        print ("Predicted Value from Model Function")
        ret = model_function_summit({
            'm': point['m'],
            'n': point['n'],
            'mb': point['mb'],
            'nb': point['nb'],
            'lg2npernode': point['lg2npernode'],
            'p': point['p']})
        return (ret) #ret['r']

    elif point['machine'] == 'mymac':
        print ("Predicted Value from Model Function")
        ret = model_function_mac({
            'm': point['m'],
            'n': point['n'],
            'mb': point['mb'],
            'nb': point['nb'],
            'lg2npernode': point['lg2npernode'],
            'p': point['p']})
        return (ret) #ret['r']

    else: # Cori
        ######################################### 
        ##### constants defined in TuningProblem
        nodes = point['nodes']
        cores = point['cores']
        bunit = point['bunit']	
        #########################################

        m = point['m']
        n = point['n']
        mb = point['mb']*bunit
        nb = point['nb']*bunit
        p = point['p']
        npernode = 2**point['lg2npernode']
        nproc = nodes*npernode
        nthreads = int(cores / npernode)  

        # this becomes useful when the parameters returned by TLA_II do not respect the constraints
        if(nproc == 0 or p == 0 or nproc < p):
            print('Warning: wrong parameters for objective function!!!')
            return 1e12
        q = int(nproc / p)
        nproc = p*q
        params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]

        print(params, ' scalapack starts ') 
        elapsedtime = pdqrdriver(params, niter=2, JOBID=JOBID)
        print(params, ' scalapack time: ', elapsedtime)

        return elapsedtime

def cst1(mb,p,m,bunit):
    return mb*bunit * p <= m
def cst2(nb,lg2npernode,n,p,nodes,bunit):
    return nb * bunit * nodes * 2**lg2npernode <= n * p
def cst3(lg2npernode,p,nodes):
    return nodes * 2**lg2npernode >= p

def main():

    global JOBID

    global model_function

    (model_function_summit, machine_configuration_string) = LoadSurrogateModelFunction({'name':'summit'})
    model_function_mac = LoadSurrogateModelFunction({'name':'mac'})

    # Parse command line arguments
    args = parse_args()

    mmax = args.mmax
    nmax = args.nmax
    ntask = args.ntask
    nprocmin_pernode = args.nprocmin_pernode
    nrun = args.nrun
    truns = args.truns
    tla = args.tla
    JOBID = args.jobid
    TUNER_NAME = args.optimization

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    os.system("mkdir -p scalapack-driver/bin/%s;" %(machine))
    DRIVERFOUND=False
    INSTALLDIR=os.getenv('GPTUNE_INSTALL_PATH')
    DRIVER = os.path.abspath(__file__ + "/../../../build/pdqrdriver")
    if(os.path.exists(DRIVER)):
        DRIVERFOUND=True
    elif(INSTALLDIR is not None):
        DRIVER = INSTALLDIR+"/gptune/pdqrdriver"
        if(os.path.exists(DRIVER)):
            DRIVERFOUND=True
    else:
        for p in sys.path:
            if("gptune" in p):
                DRIVER=p+"/pdqrdriver"
                if(os.path.exists(DRIVER)):
                    DRIVERFOUND=True
                    break
    
    if(DRIVERFOUND == True):
        os.system("cp %s scalapack-driver/bin/%s/.;" %(DRIVER,machine))
    else:
        raise Exception(f"pdqrdriver cannot be located. Try to set env variable GPTUNE_INSTALL_PATH correctly.")

    nprocmax = nodes*cores

    bunit=8     # the block size is multiple of bunit
    mmin=128
    nmin=128

    m = Integer(mmin, mmax, transform="normalize", name="m")
    n = Integer(nmin, nmax, transform="normalize", name="n")
    mb = Integer(1, 16, transform="normalize", name="mb")
    nb = Integer(1, 16, transform="normalize", name="nb")
    lg2npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="lg2npernode")
    p = Integer(1, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, lg2npernode, p])
    OS = Space([r])
    
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    constants={"nodes":nodes,"cores":cores,"bunit":bunit}
    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class'] = 'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)

    seed(1)
    if ntask == 1:
        giventask = [[mmax,nmax]]
    elif ntask == 2:
        giventask = [[4000,4000],[1000,1000]] #[[mmax,nmax],[int(mmax/2),int(nmax/2)]]
    else:
        giventask = [[randint(mmin,mmax),randint(nmin,nmax)] for i in range(ntask)]

    ntask=len(giventask)
    
    data = Data(problem)

    gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))

    """ Building MLA with the given list of tasks """
    NI = len(giventask)
    NS = nrun
    (data, model, stats) = gt.MLA(NS=NS, Tgiven=giventask, NI=NI, NS1=max(NS//2, 1))
    #(data, model, stats) = gt.MLA_LoadModel(NS=10, Tgiven=giventask)
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
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments    
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-tla', type=int, default=0, help='Whether perform TLA after MLA when optimization is GPTune')    
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
