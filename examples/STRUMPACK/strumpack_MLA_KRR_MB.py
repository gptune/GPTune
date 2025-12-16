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
from GPTune.callhpbandster import HpBandSter, HpBandSter_bandit
import math
import openturns as ot

################################################################################
def objectives(point):                  # should always use this name for user-defined objective function
    
    ######################################### 
    ##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']    
    npernode = point['npernode']        
    #########################################

    datafile = point['datafile']
    h = 10**point['h']
    Lambda = 10**point['Lambda']
    bmin = point['bmin']
    bmax = point['bmax']
    eta = point['eta']
    
    # map budget to fidelity, i.e., percentage of training data
    def budget_map(b, nmin=0.1, nmax=1):
        k = (nmax-nmin)/(bmax-bmin)
        m = nmax-bmax*k
        if b == bmin:
            return nmin
        elif b == bmax:
            return nmax
        else:
            return k * b + m

    try:
        budget = point['budget']
        fidelity = budget_map(budget)
    except:
        fidelity = budget_map(bmax)
    
    
    nproc = nodes*npernode
    nthreads = int(cores / npernode)
    
    params = ['datafile', datafile, 'h', h,'Lambda', Lambda, 'fidelity', fidelity, 'nthreads', nthreads, 'npernode', npernode, 'nproc',nproc]
    RUNDIR = os.path.abspath(__file__ + "/../STRUMPACK/examples")
    INPUTDIR = os.path.abspath(__file__ + "/../STRUMPACK/examples")
    TUNER_NAME = os.environ['TUNER_NAME']
    

    """ pass some parameters through environment variables """    
    info = MPI.Info.Create()
    envstr= 'OMP_NUM_THREADS=%d\n' %(nthreads)   
    print('OMP_NUM_THREADS=%d\n' %(nthreads))   
    info.Set('env',envstr)
    info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    
    # datafile = "data/susy_10Kn"
    # h = 0.1
    # Lambda = 3.11
    degree = 1    # only used when kernel=ANOVA (degree<=d) in KernelRegressionMPI.py 

    """ use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
    # print('exec', "%s/testPoisson3dMPIDist"%(RUNDIR), 'args', ['%s'%(gridsize), '1', '--sp_reordering_method', '%s'%(sp_reordering_method),'--sp_matching', '0','--sp_compression', '%s'%(sp_compression1),'--sp_nd_param', '%s'%(sp_nd_param),'--sp_compression_min_sep_size', '%s'%(sp_compression_min_sep_size),'--sp_compression_min_front_size', '%s'%(sp_compression_min_front_size),'--sp_compression_leaf_size', '%s'%(sp_compression_leaf_size)]+extra_str, 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads))
    comm = MPI.COMM_SELF.Spawn("%s/KernelRegressionMPI.py"%(RUNDIR), args=['%s/%s'%(INPUTDIR,datafile), '%s'%(h),'%s'%(Lambda),'%s'%(degree), '%s'%(fidelity)], maxprocs=nproc,info=info)

    """ gather the return value using the inter-communicator """                                                                    
    tmpdata = np.array([0],dtype=np.float64)
    comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
    comm.Disconnect()    

    retval = tmpdata[0]
    print(params, ' krr prediction accuracy (%): ', retval)


    return [-retval] 
    
    
def main():

    # Parse command line arguments
    args   = parse_args()

    # Extract arguments
    ntask = args.ntask
    npernode = args.npernode
    optimization = args.optimization
    nrun = args.nrun
    bmin = args.bmin
    bmax = args.bmax
    eta = args.eta
    Nloop = args.Nloop
    restart = args.restart   
    expid = args.expid 
    TUNER_NAME = args.optimization
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    ot.RandomGenerator.SetSeed(args.seed)
    print(args)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME


    datafiles = [f"data/{args.dataset}"]
    # datafiles = ["data/branin"]

    # Task input parameters
    datafile    = Categoricalnorm(datafiles, transform="onehot", name="datafile")

    # Tuning parameters
    h =  Real(-10, 10, transform="normalize", name="h")
    Lambda =  Real(-10, 10, transform="normalize", name="Lambda")
    # npernode     = Integer(int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")

    result   = Real(0 , float("Inf"),name="r")
    IS = Space([datafile])
    # PS = Space([h,Lambda,npernode])
    PS = Space([h,Lambda])
    OS = Space([result])
    constraints = {}
    models = {}
    
    constants={"nodes":nodes,"cores":cores,"npernode":npernode,"bmin":bmin,"bmax":bmax,"eta":eta}
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
    options['model_class '] = 'Model_LCM' # 'Model_GPy_LCM'
    options['verbose'] = False
 
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
    Btotal = int(np.dot(np.array(NSs_all), np.array(budget_all))/options['budget_max'] * Nloop) # total number of evaluations at highest budget -- used for single-fidelity tuners
    print(f"bmin = {bmin}, bmax = {bmax}, eta = {eta}, smax = {smax}")
    print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)    
    print("total number of samples: ", Ntotal)
    print("total number of evaluations at highest budget: ", Btotal)
    print()
    
    options.validate(computer = computer)
    
    
    # """ Building MLA with the given list of tasks """

    giventask = [[f"data/{args.dataset}"]]        
    data = Data(problem)
    NI = ntask


    
    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        NS = Btotal
        if args.nrun > 0:
            NS = args.nrun
        NS1 = max(NS//2, 1)
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=NS1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        results_file = open(f"KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
        results_file.write(f"Tuner: {TUNER_NAME}\n")
        results_file.write(f"stats: {stats}\n")
        """ Print all input and parameter samples """    
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            results_file.write(f"tid: {tid:d}\n")
            results_file.write(f"    matrix:{data.I[tid][0]:s}\n")
            # results_file.write(f"    Ps {data.P[tid]}\n")
            results_file.write(f"    Os {data.O[tid].tolist()}\n")
            # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
        results_file.close()
    
    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        if args.nrun > 0:
            NS = args.nrun
        NS1 = max(NS//2, 1)
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        results_file = open(f"KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
        results_file.write(f"Tuner: {TUNER_NAME}\n")
        results_file.write(f"stats: {stats}\n")
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid][:NS])
            print("    Os ", data.O[tid][:NS])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid][:NS])], 'Oopt ', -min(data.O[tid][:NS])[0], 'nth ', np.argmin(data.O[tid][:NS]))
            results_file.write(f"tid: {tid:d}\n")
            results_file.write(f"    matrix:{data.I[tid][0]:s}\n")
            # results_file.write(f"    Ps {data.P[tid][:NS]}\n")
            results_file.write(f"    Os {data.O[tid][:NS].tolist()}\n")
            # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
        results_file.close()
         
    if(TUNER_NAME=='TPE'):
        NS = Btotal
        if args.nrun > 0:
            NS = args.nrun
        NS1 = max(NS//2, 1)
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="HpBandSter", niter=1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        results_file = open(f"KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
        results_file.write(f"Tuner: {TUNER_NAME}\n")
        results_file.write(f"stats: {stats}\n")
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d"%(tid))
            print("    matrix:%s"%(data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', -min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            results_file.write(f"tid: {tid:d}\n")
            results_file.write(f"    matrix:{data.I[tid][0]:s}\n")
            # results_file.write(f"    Ps {data.P[tid]}\n")
            results_file.write(f"    Os {data.O[tid].tolist()}\n")
            # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
        results_file.close()
        
    if(TUNER_NAME=='GPTuneBand'):
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, options=options)
        (data, stats, data_hist)=gt.MB_LCM(NLOOP = Nloop, Tgiven = giventask)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        results_file = open(f"KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
        results_file.write(f"Tuner: {TUNER_NAME}\n")
        results_file.write(f"stats: {stats}\n")
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix:%s"%(data.I[tid][0]))
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
            print('    Popt ', Popt, 'Oopt ', -min(data.O[tid])[0], 'nth ', nth, 'nth-bandit (s, nth) = ', (arm_opt, idx))
            results_file.write(f"tid: {tid:d}\n")
            results_file.write(f"    matrix:{data.I[tid][0]:s}\n")
            # results_file.write(f"    Ps {data.P[tid]}\n")
            results_file.write(f"    Os {data.O[tid].tolist()}\n")
            # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
        results_file.close()
    
    # multi-fidelity version                
    if(TUNER_NAME=='hpbandster'):
        NS = Ntotal
        (data,stats)=HpBandSter_bandit(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        results_file = open(f"KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt", "a")
        results_file.write(f"Tuner: {TUNER_NAME}\n")
        results_file.write(f"stats: {stats}\n")
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix:%s"%(data.I[tid][0]))
            # print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            # print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            max_budget = 0.
            Oopt = 99999
            Popt = None
            nth = None
            for idx, (config, out) in enumerate(zip(data.P[tid], data.O[tid].tolist())):
                for subout in out[0]:
                    budget_cur = subout[0]
                    if budget_cur > max_budget:
                        max_budget = budget_cur
                        Oopt = subout[1]
                        Popt = config
                        nth = idx
                    elif budget_cur == max_budget:
                        if subout[1] < Oopt:
                            Oopt = subout[1]
                            Popt = config
                            nth = idx                    
            print('    Popt ', Popt, 'Oopt ', -Oopt, 'nth ', nth)
            results_file.write(f"tid: {tid:d}\n")
            results_file.write(f"    matrix:{data.I[tid][0]:s}\n")
            # results_file.write(f"    Ps {data.P[tid]}\n")
            results_file.write(f"    Os {data.O[tid].tolist()}\n")
            # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
        results_file.close()



def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments

    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-npernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-dataset', type=str,default='susy_10Kn',help='dataset')
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=-1, help='Number of runs per task')
    parser.add_argument('-bmin', type=int, default=1,  help='minimum fidelity for a bandit structure')
    parser.add_argument('-bmax', type=int, default=8, help='maximum fidelity for a bandit structure')
    parser.add_argument('-eta', type=int, default=2, help='base value for a bandit structure')
    parser.add_argument('-Nloop', type=int, default=1, help='number of GPTuneBand loops')
    parser.add_argument('-restart', type=int, default=1, help='number of GPTune MLA restart')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-expid', type=str, default='-', help='run id for experiment')


    args   = parser.parse_args()
    return args


if __name__ == "__main__":
 
    main()

