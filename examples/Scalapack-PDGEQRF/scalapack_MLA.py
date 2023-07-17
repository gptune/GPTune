#! /usr/bin/env python3

"""
Example of invocation of this script:

mpirun -n 1 python scalapack_MLA.py -mmax 5000 -nmax 5000 -nprocmin_pernode 1 -ntask 5 -nrun 10 -jobid 0 -tla_I 0 -tla_II 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -jobid is optional. You can always set it to 0.
    -tla_I is whether TLA_I is used after MLA
    -tla_II is whether TLA_II is used after MLA
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
from gptune import * # import all

import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import time
import math


################################################################################

''' The objective function required by GPTune. '''
# should always use this name for user-defined objective function
def objectives(point):

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

    # Parse command line arguments
    args = parse_args()

    mmax = args.mmax
    nmax = args.nmax
    ntask = args.ntask
    nprocmin_pernode = args.nprocmin_pernode
    nrun = args.nrun
    truns = args.truns
    tla_I = args.tla_I
    tla_II = args.tla_II
    JOBID = args.jobid
    TUNER_NAME = args.optimization
    ##### YL: the following shouldn't be hardcoded as this example always works on one machine. TLA across machines can use CrowdTuning/ScaLAPACK-PDGEQRF
    # tuning_metadata = {
    #     "tuning_problem_name": "PDGEQRF",
    #     "machine_configuration": {
    #         "machine_name": "mac",
    #         "intel": {
    #             "nodes": 1,
    #             "cores": 8
    #         }
    #     },
    #     "software_configuration": {
    #         "openmpi": {
    #             "version_split": [4,1,5]
    #         },
    #         "scalapack": {
    #             "version_split": [2,2,0]
    #         },
    #         "gcc": {
    #             "version_split": [13,1,0]
    #         }
    #     },
    #     "loadable_machine_configurations": {
    #         "mac" : {
    #             "intel": {
    #                 "nodes":1,
    #                 "cores":8
    #             }
    #         }
    #     },
    #     "loadable_software_configurations": {
    #         "openmpi": {
    #             "version_from":[4,1,5],
    #             "version_to":[5,0,0]
    #         },
    #         "scalapack":{
    #             "version_split":[2,2,0]
    #         },
    #         "gcc": {
    #             "version_split": [13,1,0]
    #         }
    #     }
    # }
    tuning_metadata=None

#    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
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
    historydb = HistoryDB(meta_dict=tuning_metadata)
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
        giventask = [[mmax,nmax],[int(mmax/2),int(nmax/2)]]
    else:
        giventask = [[randint(mmin,mmax),randint(nmin,nmax)] for i in range(ntask)]
    # # giventask = [[2000, 2000]]
    # giventask = [[177, 1303],[367, 381],[1990, 1850],[1123, 1046],[200, 143],[788, 1133],[286, 1673],[1430, 512],[1419, 1320],[622, 263] ]
    # giventask = [[177, 1303],[367, 381]]
    ntask=len(giventask)

    data = Data(problem)
    if(TUNER_NAME=='GPTune'):

        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

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

        if(tla_I==1):
            """ Call TLA for 2 new tasks using the constructed LCM model"""

            # the data object initialized to run transfer learning as a new autotuning run
            data = Data(problem)
            historydb=HistoryDB(meta_dict=tuning_metadata)
            gt = GPTune(problem, computer=computer, data=data, options=options,historydb=historydb, driverabspath=os.path.abspath(__file__))

            # load source function evaluation data
            def LoadFunctionEvaluations(Tsrc):
                function_evaluations = [[] for i in range(len(Tsrc))]
                with open ("gptune.db/PDGEQRF.json", "r") as f_in:
                    for func_eval in json.load(f_in)["func_eval"]:
                        task_parameter = [func_eval["task_parameter"]["m"], func_eval["task_parameter"]["n"]]
                        if task_parameter in Tsrc:
                            function_evaluations[Tsrc.index(task_parameter)].append(func_eval)
                return function_evaluations

            options["TLA_method"] = "LCM"
            options["model_class"] = "Model_GPy_LCM"
            options.validate(computer=computer)
            data = Data(problem)
            gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
            newtask = [[400, 500]]
            (data, modeler, stats) = gt.TLA_I(NS=nrun, Tnew=newtask, source_function_evaluations=LoadFunctionEvaluations(giventask))

            """ Print all input and parameter samples """
            for tid in range(len(data.I)):
                print("tid: %d" % (tid))
                print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
                print("    Ps ", data.P[tid])
                print("    Os ", data.O[tid].tolist())
                print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if(tla_II==1):
            """ Call TLA for 2 new tasks using the constructed LCM model"""

            # the data object initialized to run transfer learning as a new autotuning run
            data = Data(problem)
            historydb=HistoryDB(meta_dict=tuning_metadata)
            gt = GPTune(problem, computer=computer, data=data, options=options,historydb=historydb, driverabspath=os.path.abspath(__file__))

            # load source function evaluation data
            def LoadFunctionEvaluations(Tsrc):
                function_evaluations = [[] for i in range(len(Tsrc))]
                with open ("gptune.db/PDGEQRF.json", "r") as f_in:
                    for func_eval in json.load(f_in)["func_eval"]:
                        task_parameter = [func_eval["task_parameter"]["m"], func_eval["task_parameter"]["n"]]
                        if task_parameter in Tsrc:
                            function_evaluations[Tsrc.index(task_parameter)].append(func_eval)
                return function_evaluations

            newtask = [[400, 500], [800, 600]]
            (aprxopts, objval, stats) = gt.TLA_II(Tnew=newtask, Tsrc=giventask, source_function_evaluations=LoadFunctionEvaluations(giventask))
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])


    if(TUNER_NAME=='opentuner'):
        NI = len(giventask)
        NS = nrun
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
        NI = len(giventask)
        NS = nrun

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
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments    
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-tla_I', type=int, default=0, help='Whether perform TLA_I after MLA when optimization is GPTune')
    parser.add_argument('-tla_II', type=int, default=0, help='Whether perform TLA_II after MLA when optimization is GPTune')
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
