#! /usr/bin/env python

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

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import sys
import os
import mpi4py
from mpi4py import MPI
import numpy as np
import time
import argparse
from callopentuner import OpenTuner
from callhpbandster import HpBandSter, HpBandSter_bandit
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

# from GPTune import *

################################################################################

# Define Problem
def objectives(point):
    nodes = point['nodes']
    cores = point['cores']
    nstepmax = point['nstepmax']
    nstepmin = point['nstepmin']
    bmin = point['bmin']
    bmax = point['bmax']
    eta = point['eta']
    
    nprocmax = nodes*cores

    def budget_map(b, nmin=10, nmax=100):
        k1 = (nmax-nmin)/(bmax-bmin)
        b1 = nmin - k1
        assert k1 * bmax + b1 == nmax
        return int(k1 * b + b1) 
        # return int(45*(np.log(b)/np.log(eta)) + 10)
        # return int(10*b)
    try:
        budget = point['budget']
        nstep = budget_map(budget,nstepmin,nstepmax)
    except:
        nstep = budget_map(bmax,nstepmin,nstepmax)

    # COLPERM = point['COLPERM']
    # ROWPERM = point['ROWPERM']    
    
    COLPERM = '4'
    ROWPERM = '2'
    mx = point['mx']
    my = point['my']
    lphi = point['lphi']
    # nstep = point['nstep']
    # nprows = 2**point['nprows']
    # nproc = 2**point['nproc']
    # nproc = 32
    NSUP = point['NSUP']
    NREL = point['NREL']
    nbx = point['nbx']
    nby = point['nby']
    # nblock     = int(nprocmax/nproc)
    # npcols     = int(nproc/ nprows)
    params = ['mx',mx,'my',my,'lphi',lphi,'nstep',nstep,'ROWPERM', ROWPERM, 'COLPERM', COLPERM, 'NSUP', NSUP, 'NREL', NREL, 'nbx', nbx, 'nby', nby]

    # # INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")

    nthreads   = 1


    """ pass some parameters through environment variables """	


    info = MPI.Info.Create()
    envstr= 'OMP_NUM_THREADS=1\n'
    envstr+= 'NREL=%d\n' %(NREL)   
    envstr+= 'NSUP=%d\n' %(NSUP)   
    info.Set('env',envstr)

    #####################################
    ####### npernode is very important, without setting it the application can be much slower
    info.Set('npernode','%d'%(cores)) # flat MPI # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
    #####################################

    fin = open("./nimrod_template.in","rt")
    fout = open("./nimrod.in","wt")

    for line in fin:
        #read replace the string and write to output file
        if(line.find("iopts(3)")!=-1):
            fout.write("iopts(3)= %s\n"%(ROWPERM))
        elif(line.find("iopts(4)")!=-1):
            fout.write("iopts(4)= %s\n"%(COLPERM))    
        elif(line.find("lphi")!=-1):
            fout.write("lphi= %s\n"%(lphi))    
        elif(line.find("nlayers")!=-1):
            fout.write("nlayers= %s\n"%(int(np.floor(2**lphi/3.0)+1)))  	
        elif(line.find("mx")!=-1):
            fout.write("mx= %s\n"%(2**mx))
        elif(line.find("nstep")!=-1):
            fout.write("nstep= %s\n"%(nstep))  			  
        elif(line.find("my")!=-1):
            fout.write("my= %s\n"%(2**my))   
        elif(line.find("nxbl")!=-1):
            fout.write("nxbl= %s\n"%(int(2**mx/2**nbx)))  
        elif(line.find("nybl")!=-1):
            fout.write("nybl= %s\n"%(int(2**my/2**nby)))  									  						        
        else:
            fout.write(line)
    #close input and output files
    fin.close()
    fout.close()


    nlayers=int(np.floor(2**lphi/3.0)+1)
    nproc = int(nprocmax/nlayers)*nlayers
    if(nprocmax<nlayers):
        print('nprocmax', nprocmax, 'nlayers', nlayers, 'decrease lphi!')
        raise Exception("nprocmax<nlayers")
    if(nproc>int(2**mx/2**nbx)*int(2**my/2**nby)*int(np.floor(2**lphi/3.0)+1)): # nproc <= nlayers*nxbl*nybl
        nproc = int(2**mx/2**nbx)*int(2**my/2**nby)*int(np.floor(2**lphi/3.0)+1) 

    os.system("./nimset")


    nrep=1 #3
    hist=[]
    for i in range(nrep):
        """ use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
        print('exec', "./nimrod_spawn", 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL))
        comm = MPI.COMM_SELF.Spawn("./nimrod_spawn", maxprocs=nproc,info=info)
        """ gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """																	
        tmpdata = np.array([0,0,0,0,0],dtype=np.float64)
        comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.DOUBLE],op=MPI.MAX,root=mpi4py.MPI.ROOT) 
        comm.Disconnect()
        time.sleep(5.0)
        hist.append(tmpdata)
        print(params, ' nimrod time (trial) -- loop:', tmpdata[0],'slu: ', tmpdata[1],'factor: ', tmpdata[2], 'iter: ', tmpdata[3], 'total: ', tmpdata[4])
    
    tmpdata = min(hist, key=lambda x: x[0])
    retval = tmpdata[0]
    print(params, ' nimrod time -- loop:', tmpdata[0],'slu: ', tmpdata[1],'factor: ', tmpdata[2], 'iter: ', tmpdata[3], 'total: ', tmpdata[4])

    return retval 
















def main():

    
    args = parse_args()
    ntask = args.ntask
    Nloop = args.Nloop
    bmin = args.bmin
    bmax = args.bmax
    eta = args.eta

    TUNER_NAME = args.optimization
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    nstepmax = args.nstepmax
    nstepmin = args.nstepmin
    
    os.environ['TUNER_NAME'] = TUNER_NAME



    # Input parameters
    # ROWPERM   = Categoricalnorm (['1', '2'], transform="onehot", name="ROWPERM")
    # COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
    # nprows    = Integer     (0, 5, transform="normalize", name="nprows")
    # nproc    = Integer     (5, 6, transform="normalize", name="nproc")
    NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
    NREL      = Integer     (10, 40, transform="normalize", name="NREL")
    nbx      = Integer     (1, 3, transform="normalize", name="nbx")	
    nby      = Integer     (1, 3, transform="normalize", name="nby")	

    time   = Real        (float("-Inf") , float("Inf"), transform="normalize", name="time")

    # nstep      = Integer     (3, 15, transform="normalize", name="nstep")
    lphi      = Integer     (2, 3, transform="normalize", name="lphi")
    mx      = Integer     (5, 6, transform="normalize", name="mx")
    my      = Integer     (7, 8, transform="normalize", name="my")

    IS = Space([mx,my,lphi])
    # PS = Space([ROWPERM, COLPERM, nprows, nproc, NSUP, NREL])
    # PS = Space([ROWPERM, COLPERM, NSUP, NREL, nbx, nby])
    PS = Space([NSUP, NREL, nbx, nby])
    OS = Space([time])
    cst1 = "NSUP >= NREL"
    constraints = {"cst1" : cst1}
    models = {}
    constants={"nodes":nodes,"cores":cores,"nstepmin":nstepmin,"nstepmax":nstepmax,"bmin":bmin,"bmax":bmax,"eta":eta}

    """ Print all input and parameter samples """	
    print(IS, PS, OS, constraints, models)

    BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_haswell_gnu_openmpi/bin")
    # BINDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_knl_gnu_openmpi/bin")
    RUNDIR = os.path.abspath("/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimrod_input")
    os.system("cp %s/nimrod.in ./nimrod_template.in"%(RUNDIR))
    os.system("cp %s/fluxgrid.in ."%(RUNDIR))
    os.system("cp %s/g163518.03130 ."%(RUNDIR))
    os.system("cp %s/p163518.03130 ."%(RUNDIR))
    os.system("cp %s/nimset ."%(RUNDIR))
    os.system("cp %s/nimrod ./nimrod_spawn"%(BINDIR))



    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)  

    """ Set and validate options """	
    options = Options()

    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1
    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16
    # options['mpi_comm'] = None
    # options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_LCM' if args.LCMmodel == 'LCM' else 'Model_GPy_LCM' # Model_GPy_LCM or Model_LCM
    options['verbose'] = True
    options['sample_class'] = 'SampleLHSMDU'
    options['sample_algo'] = 'LHS-MDU'
    options.validate(computer=computer)

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
    # giventask = [[1.0], [5.0], [10.0]]
    # giventask = [[1.0], [1.2], [1.3]]
    giventask = [[6,8,2]]
    Pdefault = [128,20,2,2]
    # t_end = args.t_end
    # giventask = [[i] for i in np.arange(1, t_end, (t_end-1)/ntask).tolist()] # 10 tasks
    # giventask = [[i] for i in np.arange(1.0, 6.0, 1.0).tolist()] # 5 tasks
    NI=len(giventask)
    assert NI == ntask # make sure number of tasks match
    
    np.set_printoptions(suppress=False, precision=4)
    if(TUNER_NAME=='GPTuneBand'):
        NS = Nloop
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
        (data, stats, data_hist)=gt.MB_LCM(NS = Nloop, Igiven = giventask, Pdefault=Pdefault)
        print("Tuner: ", TUNER_NAME)
        print("Sampler class: ", options['sample_class'])
        print("Model class: ", options['model_class'])
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
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
        print("stats: ", stats)
        print("Sampler class: ", options['sample_class'], "Sample algo:", options['sample_algo'])
        print("Model class: ", options['model_class'])
        if options['model_class'] == 'Model_LCM' and NI > 1:
            print("Get correlation metric ... ")
            C = model[0].M.kern.get_correlation_metric()
            print("The correlation matrix C is \n", C)
        elif options['model_class'] == 'Model_GPy_LCM' and NI > 1:
            print("Get correlation metric ... ")
            C = model[0].get_correlation_metric(NI)
            print("The correlation matrix C is \n", C)

        
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], f'Oopt  {min(data.O[tid])[0]:.3f}', 'nth ', np.argmin(data.O[tid]))
            
    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid][:NS])], 'Oopt ', min(data.O[tid][:NS])[0], 'nth ', np.argmin(data.O[tid][:NS]))
            
    if(TUNER_NAME=='hpbandster'):
        NS = Btotal
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            
    
    if(TUNER_NAME=='TPE'):
        NS = Ntotal
        (data,stats)=callhpbandster_bandit.HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
            print("    Ps ", data.P[tid])
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
            print('    Popt ', Popt, 'Oopt ', Oopt, 'nth ', nth)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bmin', type=int, default=1, help='budget min')   
    parser.add_argument('-bmax', type=int, default=2, help='budget max')   
    parser.add_argument('-eta', type=int, default=2, help='eta')   
    parser.add_argument('-nstepmax', type=int, default=-1, help='maximum number of time steps')   
    parser.add_argument('-nstepmin', type=int, default=-1, help='minimum number of time steps')   
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=-1, help='total application runs')
    parser.add_argument('-LCMmodel', type=str, default='LCM', help='choose from LCM models: LCM or GPy_LCM')
    parser.add_argument('-Nloop', type=int, default=1, help='Number of outer loops in multi-armed bandit per task')
    # parser.add_argument('-sample_class', type=str,default='SampleOpenTURNS',help='Supported sample classes: SampleLHSMDU, SampleOpenTURNS')
    args = parser.parse_args()
    
    return args   


if __name__ == "__main__":
    main()