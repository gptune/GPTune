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



def initialize(m, n, nodes, cores, nruns, truns, machine, JOBID=0):

    ROOTDIR = os.path.abspath(__file__ + "/../scalapack-driver/")
    EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "exp/%s/"%(machine), JOBID))
    if (JOBID != 0):
        RUNDIR = os.path.abspath(os.path.join(EXPDIR, "m_%d_n_%d_nodes_%d_cores_%d_jobid_%d_nruns_%d"%(m, n, nodes, cores, JOBID, nruns)))
    else:
        RUNDIR = os.path.abspath(os.path.join(EXPDIR, str(os.getpid())))

    # try:
        # os.makedirs(EXPDIR)
    # except:
        # pass
    # try:
        # os.makedirs(RUNDIR)
    # except:
        # pass

    return RUNDIR

# def myobjfun(m, n, mb, nb, nproc, p):
def objective(point):                  # should always use this name for user-defined objective function
    m = point['m']
    n = point['n']
    mb = point['mb']
    nb = point['nb']
    nproc = point['nproc']
    p = point['p']

    
#        return np.random.rand(1)
    if(nproc==0):
        return 1000.0
    if(p==0):
        return 1000.0
    nth   = int((nodes * cores-2) / nproc) # YL: there are at least 2 cores working on other stuff
    q     = int(nproc / p)


# [("fac", 'U10'), ("m", int), ("n", int), ("nodes", int), ("cores", int), ("mb", int), ("nb", int), ("nth", int), ("nproc", int), ("p", int), ("q", int), ("thresh", float)]
    params = [('QR', m, n, nodes, cores, mb, nb, nth, nproc, p, q, 1.)]
    # print(params,' in myobjfun')

    repeat = True
#        while (repeat):
#            try:
    elapsedtime = pdqrdriver(params, niter = 3)
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


#    RUNDIR = initialize(0, 0, nodes, cores, nruns, truns, "mogpo", machine, JOBID=JOBID)
    # RUNDIR = os.path.abspath(__file__ + "/../scalapack-driver/exp/%s/%d"%(machine, os.getpid()))
    # try:
    #     os.makedirs(RUNDIR)
    # except:
    #     pass


	
# YL: for the spaces, the following datatypes are supported: 
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")  	
		
	
	
    m     = Integer (128 , mmax, transform="normalize", name="m")
    n     = Integer (128 , nmax, transform="normalize", name="n")
    mb    = Integer (1 , 128, transform="normalize", name="mb")
    nb    = Integer (1 , 128, transform="normalize", name="nb")
    nproc = Integer (nodes, nodes*cores, transform="normalize", name="nproc")
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
    cst3 = "%d * %d"%(nodes, cores) + ">= nproc+2"  # YL: there are at least 2 cores working on other stuff
    cst4 = "nproc >= p" # intrinsically implies "p <= nproc"

    constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3, "cst4" : cst4}
    # constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3}

    models = {}

    print(IS, PS, OS, constraints, models)

    problem = TuningProblem(IS, PS, OS, objective, constraints, None)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)  

    options = Options()
    options['model_processes'] = 1
    options['model_threads'] = 1
    options['model_restarts'] = 1
    options['search_multitask_processes'] = 1
    options['model_restart_processes'] = 1
    options['distributed_memory_parallelism'] = True
    options['shared_memory_parallelism'] = False
    options['mpi_comm'] = None
    options['model_class '] = 'Model_LCM'
    data = Data(problem)
    gt = GPTune(problem, computer = computer, data = data, options = options)



    NI = ntask
    NS = nruns

    (data, model) = gt.MLA(NS=NS, NI=NI, NS1 = max(NS//2,1))

    for tid in range(NI):
        print("tid: %d"%(tid))
        print("    m:%d n:%d"%(data.T[tid][0], data.T[tid][1]))
        print("    Xs ", data.X[tid])
        print("    Ys ", data.Y[tid])
        print('    Xopt ', data.X[tid][np.argmin(data.Y[tid])], 'Yopt ', min(data.Y[tid])[0])

    newtask = [[400,500],
               [1000,1200]]
    (aprxopts,objval) = gt.TLA1(newtask, nruns)
    
    for tid in range(len(newtask)):
        print("new task: %s"%(newtask[tid]))
        print('    predicted Xopt: ', aprxopts[tid], ' objval: ',objval[tid]) 	
		
		
		
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

# def define_problem(mmax, nmax, nodes, cores):

    # m     = Integer     (name="m",     range=(1 , mmax))
    # n     = Integer     (name="n",     range=(1 , nmax))
    # mb    = Integer     (name="mb",    range=(1 , mmax))
    # nb    = Integer     (name="nb",    range=(1 , nmax))
# #    nproc = Integer (name="nproc", range=(nodes , nodes*cores))
# #    p     = Integer (name="p",     range=(1 , nodes*cores))
    # nproc = Categorical (name="nproc", values=[nodes * x for x in factors(cores)])
    # p     = Categorical (name="p",     values=factors(nodes * cores))
    # r     = Real        (name="r",     range=(float("-Inf") , float("Inf")))

    # TS = Space(params=[m, n])
    # IS = Space(params=[mb, nb, nproc, p])
    # OS = Space(params=[r])

    # cst1 = "mb * p <= m"
    # cst2 = "nb * nproc <= n * p"
    # #cst3 = "int(nodes * cores / nproc) == (nodes * cores / nproc)"
    # cst3 = "%d * %d"%(nodes, cores) + "% nproc == 0"
    # cst4 = "nproc > p and nproc % p == 0" # intrinsically implies "p <= nproc"

    # constraints = {"cst1" : cst1, "cst2" : cst2, "cst3" : cst3, "cst4" : cst4}

    # models = {}

    # return (TS, IS, OS, constraints, models)

# def main_batch():

#     global ROOTDIR

#     (mmax, nmax, ntask, nodes, cores, machine, optimization, nruns, truns, JOBID, STEPID, PHASE) = parse_args()

#     (TS, IS, OS, constraints, models) = define_problem(mmax, nmax, nodes, cores)

# #    RUNDIR = initialize(0, 0, nodes, cores, nruns, truns, "mogpo", machine, JOBID=JOBID)
#     RUNDIR = os.path.abspath(__file__ + "/../scalapack-driver/exp/%s/%d"%(machine, os.getpid()))
#     try:
#         os.makedirs(RUNDIR)
#     except:
#         pass
#     EXPDIR = os.path.abspath(__file__ + "/../exp/ztune")
#     try:
#         os.makedirs(EXPDIR)
#     except:
#         pass

#     os.environ["OPENBLAS_NUM_THREADS"] = "1"

#     if (PHASE == 0):

#         z = ZTune(TS, IS, OS, objfun = None, cstrs = constraints, objmdls = models, name = "ScaLAPACK Tuner")
#         print(TS, IS, OS)

#         NT = ntask
#         NS = nruns
#         ratio_init_vs_opti = 0.2
#         Q = 0

#         #modeltypes = ["pyLCM", "cLCM", "DGP"]
#         #modeltypes = ["DGP", "cLCM"]
#         modeltypes = ["cLCM"]
#         #modeltypes = ["DGP"]

#         z = {modeltype: ZTune(TS, IS, OS, objfun = None, cstrs = constraints, objmdls = {}, name = "", modeltype=modeltype) for modeltype in modeltypes} 
#         z.update({(modeltype, tid): ZTune(TS, IS, OS, objfun = None, cstrs = constraints, objmdls = {}, name = "", modeltype=modeltype) for modeltype in modeltypes for tid in range(NT)})

#         newXs = []
#         if (STEPID == 0):
#             #T = z[modeltypes[0]].generate_tasks(NT)
#             #pickle.dump(T, open(EXPDIR + "/T.pkl", 'wb'))
#             T = pickle.load(open(EXPDIR + "/T.pkl", 'rb'))
#             NS1 = min(NS - 1, max(3 * len(IS), int(NS * ratio_init_vs_opti)))
#             for key in ["cLCM"]:#z: # XXX temporary
#                 z[key].T = T
#                 if (isinstance(key, str)):
#                     z[key].X = z[key].generate_samples(T, NS1)
#                 else:
#                     z[key].X = z[key].generate_samples([T[key[1]]], NS1)
#                 newXs.append((key, z[key].X))
#         else:
#             T = pickle.load(open(EXPDIR + "/T.pkl", 'rb'))
#             for key in z:
#                 if (isinstance(key, str)):
#                     z[key].T = T
#                     z[key].X = pickle.load(open(EXPDIR + "/X_%d_%s.pkl"%(STEPID, key), 'rb'))
#                     z[key].Y = pickle.load(open(EXPDIR + "/Y_%d_%s.pkl"%(STEPID, key), 'rb'))
#                     print(key, z[key].Y)
#                     z[key].train(Q = Q)
#                     newX = z[key].search()
#                     for i in range(len(T)):
#                         z[key].X[i] = np.concatenate([z[key].X[i], newX[i]])
#                 else:
#                     z[key].T = T[key[1]].reshape((1, len(TS)))
#                     z[key].X = pickle.load(open(EXPDIR + "/X_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'rb'))
#                     z[key].Y = pickle.load(open(EXPDIR + "/Y_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'rb'))
#                     print(key, z[key].Y)
#                     z[key].train(Q = Q)
#                     newX = z[key].search()
#                     z[key].X[0] = np.concatenate([z[key].X[0], newX[0]])
#                 newXs.append((key, newX))

#         params = []
#         for (key, newX) in newXs:
#             if (isinstance(key, str)):
#                 for tid in range(len(T)):
#                     t = z[key].denormalize_tasks(T[tid])
#                     for xx in newX[tid]:
#                         print(key, xx)
#                         x = z[key].denormalize_samples(xx)
#                         print(key, x)
#                         params.append(('QR', t[0], t[1], nodes, cores, x[0], x[1], int(nodes * cores / x[2]), x[2], x[3], int(x[2] / x[3]), 1.))
# #                        params.append(('QR', t[0], t[1], nodes, cores, xx[0], xx[1], int(nodes * cores / xx[2]), xx[2], xx[3], int(xx[2] / xx[3]), 1.))  # XXX temporary
#             else:
#                 t = z[key].denormalize_tasks(T[key[1]])
#                 for xx in newX[0]:
#                     print(key, xx)
#                     x = z[key].denormalize_samples(xx)
#                     print(key, x)
#                     params.append(('QR', t[0], t[1], nodes, cores, x[0], x[1], int(nodes * cores / x[2]), x[2], x[3], int(x[2] / x[3]), 1.))

#         pickle.dump((z, T, newXs, params), open(EXPDIR + "/z_T_newXs_params_%d_cLCM.pkl"%(STEPID), 'wb'))
#         #pickle.dump((z, T, newXs, params), open(EXPDIR + "/z_T_newXs_params_%d_DGP.pkl"%(STEPID), 'wb'))
# #        elapsedtime = pdqrdriver(params, niter = 1)

#     else:

# #        (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d.pkl"%(STEPID), 'rb'))
#         (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d_cLCM.pkl"%(STEPID), 'rb'))
# #        (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d_DGP.pkl"%(STEPID), 'rb'))
# #        elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d.pkl"%(STEPID), 'rb'))
#         elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d_cLCM.pkl"%(STEPID), 'rb'))
# #        elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d_DGP.pkl"%(STEPID), 'rb'))

#         STEPID += 1
#         cpt = 0
#         for (key, newX) in newXs:
# #        for key in z: # XXX temporary
# #            if (key[0] == "cLCM"):
# #                continue
#             if (isinstance(key, str)):
#                 cpt = 0
# #                newX = newXs[0][1] # XXX temporary
# #                z[key].X = newX # XXX temporary
#                 newY = []
#                 for tid in range(len(T)):
# #                    print(len(newX[tid]))
#                     y = elapsedtime[cpt:cpt + len(newX[tid])]
#                     cpt += len(newX[tid])
#                     idxnoinf = y != np.inf
# #                    z[key].X[tid] = z[key].X[tid][idxnoinf]
#                     newY.append(np.array(y[idxnoinf]).reshape((len(idxnoinf), len(OS))))
#                 if (z[key].Y is not None):
#                     for i in range(len(T)):
#                         z[key].Y[i] = np.concatenate([z[key].Y[i], newY[i]])
#                 else:
#                     z[key].Y = newY
#                 pickle.dump(z[key].X, open(EXPDIR + "/X_%d_%s.pkl"%(STEPID, key), 'wb')) 
#                 pickle.dump(z[key].Y, open(EXPDIR + "/Y_%d_%s.pkl"%(STEPID, key), 'wb')) 
#                 cpt = 0
#             else:
# #                newX = [newXs[0][1][key[1]]] # XXX temporary
#                 print(cpt,cpt + len(newX[0]))
#                 y = elapsedtime[cpt:cpt + len(newX[0])]
#                 cpt += len(newX[0])
#                 idxnoinf = y != np.inf
# #                z[key].X = [newX[0][idxnoinf]] # XXX temporary
#                 newY = [np.array(y[idxnoinf]).reshape((len(idxnoinf), len(OS)))]
#                 if (z[key].Y is not None):
#                     z[key].Y[0] = np.concatenate([z[key].Y[0], newY[0]])
#                 else:
#                     z[key].Y = newY
#                 pickle.dump(z[key].X, open(EXPDIR + "/X_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'wb'))
#                 pickle.dump(z[key].Y, open(EXPDIR + "/Y_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'wb'))

if __name__ == "__main__":

#    os.environ['MACHINE_NAME']='cori'
#    os.environ['TUNER_NAME']='GPTune'
#    print(os.environ)
#    MACHINE_NAME = os.environ['MACHINE_NAME']
#   TUNER_NAME = os.environ['TUNER_NAME']  

 
   main_interactive()
    # main_batch()

#    else:
#
##        (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d.pkl"%(STEPID), 'rb'))
#        (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d_cLCM.pkl"%(STEPID), 'rb'))
##        (z, T, newXs, params) = pickle.load(open(EXPDIR + "/z_T_newXs_params_%d_DGP.pkl"%(STEPID), 'rb'))
##        elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d.pkl"%(STEPID), 'rb'))
#        elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d_cLCM.pkl"%(STEPID), 'rb'))
##        elapsedtime = pickle.load(open(EXPDIR + "/elapsedtime_%d_DGP.pkl"%(STEPID), 'rb'))
#
#        STEPID += 1
#        cpt = 0
#        for (key, newX) in newXs:
##        for key in z: # XXX temporary
##            if (key[0] == "cLCM"):
##                continue
#            if (isinstance(key, str)):
#                cpt = 0
##                newX = newXs[0][1] # XXX temporary
#                z[key].X = newX # XXX temporary
#                newY = []
#                for tid in range(len(T)):
#                    print(len(newX[tid]))
#                    y = elapsedtime[cpt:cpt + len(newX[tid])]
#                    cpt += len(newX[tid])
#                    idxnoinf = y != np.inf
#                    z[key].X[tid] = z[key].X[tid][idxnoinf]
#                    newY.append(np.array(y[idxnoinf]).reshape((len(z[key].X[tid]), len(OS))))
#                if (z[key].Y is not None):
#                    for i in range(len(T)):
#                        z[key].Y[i] = np.concatenate([z[key].Y[i], newY[i]])
#                else:
#                    z[key].Y = newY
#                pickle.dump(z[key].X, open(EXPDIR + "/X_%d_%s.pkl"%(STEPID, key), 'wb')) 
#                pickle.dump(z[key].Y, open(EXPDIR + "/Y_%d_%s.pkl"%(STEPID, key), 'wb')) 
#                cpt = 0
#            else:
#                newX = [newXs[0][1][key[1]]] # XXX temporary
#                print(cpt,cpt + len(newX[0]))
#                y = elapsedtime[cpt:cpt + len(newX[0])]
#                cpt += len(newX[0])
#                idxnoinf = y != np.inf
#                z[key].X = [newX[0][idxnoinf]] # XXX temporary
#                newY = [np.array(y[idxnoinf]).reshape((len(z[key].X[0]), len(OS)))]
#                if (z[key].Y is not None):
#                    z[key].Y[0] = np.concatenate([z[key].Y[0], newY[0]])
#                else:
#                    z[key].Y = newY
#                pickle.dump(z[key].X, open(EXPDIR + "/X_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'wb'))
#                pickle.dump(z[key].Y, open(EXPDIR + "/Y_%d_%s_%d.pkl"%(STEPID, key[0], key[1]), 'wb'))
#
