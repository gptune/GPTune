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



# def myobjfun(m, n, mb, nb, nproc, p):
def objective(point):                  # should always use this name for user-defined objective function
    m = point['m']
    n = point['n']
    mb = point['mb']
    nb = point['nb']
    nproc = point['nproc']
    p = point['p']

    
#        return np.random.rand(1)
    if(nproc==0 or p==0 or nproc<p):
        print('Warning: wrong parameters for objective function!!!')
        return 1e12

    nth   = int((nodes * cores-1) / nproc) # YL: there are is one proc doing spawning
    q     = int(nproc / p)


# [("fac", 'U10'), ("m", int), ("n", int), ("nodes", int), ("cores", int), ("mb", int), ("nb", int), ("nth", int), ("nproc", int), ("p", int), ("q", int), ("thresh", float)]
    params = [('QR', m, n, nodes, cores, mb, nb, nth, nproc, p, q, 1.)]
    # print(params,' in myobjfun')

    repeat = True
#        while (repeat):
#            try:
    elapsedtime = pdqrdriver(params, niter = 3, JOBID=JOBID)
    # elapsedtime = 1.0
    repeat = False
#            except:
#                print("Error in call to ScaLAPACK with parameters ", params)
#                pass

    print(params, ' scalapack time: ', elapsedtime)

    return elapsedtime 

def main_interactive():

    global ROOTDIR
    global nodes
    global cores
    global JOBID

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

    args   = parser.parse_args()

    # Extract arguments

    mmax = args.mmax
    nmax = args.nmax
    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    optimization = args.optimization
    nruns = args.nruns
    truns = args.truns
    JOBID = args.jobid
    
    
    
    os.environ['MACHINE_NAME']=machine
    os.environ['TUNER_NAME']='GPTune'
    # print(os.environ)



    os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;"%(machine, machine))


	
# YL: for the spaces, the following datatypes are supported: 
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")  	
		
	
	
    m     = Integer (128 , mmax, transform="normalize", name="m")
    n     = Integer (128 , nmax, transform="normalize", name="n")
    mb    = Integer (1 , 128, transform="normalize", name="mb")
    nb    = Integer (1 , 128, transform="normalize", name="nb")
    nproc = Integer (nodes, nodes*cores-1, transform="normalize", name="nproc") # YL: there are is one proc doing spawning
    p     = Integer (1 , nodes*cores, transform="normalize", name="p")
    r     = Real    (float("-Inf") , float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, nproc, p])
    OS = Space([r])



#    cst1 = "mb <= int(m / p) if (m / p) >= 1 else False"
#    cst2 = "nb <= int(n / int(nproc / p)) if (n / int(nproc / p)) >= 1 else False"
#    #cst3 = "int(nodes * cores / nproc) == (nodes * cores / nproc)"
#    cst3 = "int(%d * %d / nproc) == (%d * %d / nproc)"%(nodes, cores, nodes, cores)
#    cst4 = "int(nproc / p) == (nproc / p)" # intrinsically implies "p <= nproc"
    cst1 = "mb * p <= m"
    cst2 = "nb * nproc <= n * p"
    #cst3 = "int(nodes * cores / nproc) == (nodes * cores / nproc)"
    # cst3 = "%d * %d"%(nodes, cores) + ">= nproc+2"  
    cst3 = "nproc >= p" # intrinsically implies "p <= nproc"

    constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3}
    # constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3}

    models = {}

    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objective, constraints, None)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)  

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
    data = Data(problem)
    gt = GPTune(problem, computer = computer, data = data, options = options)


    # """ Building MLA with NI random tasks """
    # NI = ntask
    # NS = nruns
    # (data, model,stats) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))
    # print("stats: ",stats)
	
    """ Building MLA with the given list of tasks """	
    giventask = [[460,500],[800,690]]	
    NI = len(giventask)
    NS = nruns
    (data, model,stats) = gt.MLA(NS=NS, NI=NI, Igiven =giventask, NS1 = max(NS//2,1))
    print("stats: ",stats)
    
    pickle.dump(gt, open('MLA_nodes_%d_cores_%d_mmax_%d_nmax_%d_machine_%s_jobid_%d.pkl'%(nodes,cores,mmax,nmax,machine,JOBID), 'wb'))
	
	
	
	

    for tid in range(NI):
        print("tid: %d"%(tid))
        print("    m:%d n:%d"%(data.I[tid][0], data.I[tid][1]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid])
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Yopt ', min(data.O[tid])[0])

    newtask = [[400,500],
               [800,600]]
    (aprxopts,objval,stats) = gt.TLA1(newtask, nruns)
    print("stats: ",stats)
		
    for tid in range(len(newtask)):
        print("new task: %s"%(newtask[tid]))
        print('    predicted Popt: ', aprxopts[tid], ' objval: ',objval[tid]) 	
		
		
		
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

#    os.environ['MACHINE_NAME']='cori'
#    os.environ['TUNER_NAME']='GPTune'
#    print(os.environ)
#    MACHINE_NAME = os.environ['MACHINE_NAME']
#   TUNER_NAME = os.environ['TUNER_NAME']  

 
   main_interactive()
 
