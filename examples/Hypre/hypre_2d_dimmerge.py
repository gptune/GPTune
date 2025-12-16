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
"""
Example of invocation of this script:

mpirun -n 1 python hypre.py -nxmax 200 -nymax 200 -nzmax 200 -nprocmin_pernode 1 -ntask 20 -nrun 800

where:
    -nxmax/nymax/nzmax       maximum number of discretization size for each dimension
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask                   number of different tasks to be tuned
    -nrun                    number of calls per task
    
Description of the parameters of Hypre AMG:
Task space:
    nx:    problem size in dimension x
    ny:    problem size in dimension y
    nz:    problem size in dimension z
    cx:    diffusion coefficient for d^2/dx^2
    cy:    diffusion coefficient for d^2/dy^2
    cz:    diffusion coefficient for d^2/dz^2
    ax:    convection coefficient for d/dx
    ay:    convection coefficient for d/dy
    az:    convection coefficient for d/dz
Input space:
    Px:                processor topology, with Nproc = Px*Py*Pz where Pz is a dependent parameter
    Py:                processor topology, with Nproc = Px*Py*Pz where Pz is a dependent parameter
    Nproc:             total number of MPIs 
    strong_threshold:  AMG strength threshold
    trunc_factor:      Truncation factor for interpolation
    P_max_elmts:       Max number of elements per row for AMG interpolation
    coarsen_type:      Defines which parallel coarsening algorithm is used
    relax_type:        Defines which smoother to be used
    smooth_type:       Enables the use of more complex smoothers
    smooth_num_levels: Number of levels for more complex smoothers
    interp_type:       Defines which parallel interpolation operator is used  
    agg_num_levels:    Number of levels of aggressive coarsening
"""
import sys, os
# add GPTunde path in front of all python pkg path

sys.path.insert(0, os.path.abspath(__file__ + "/../hypre-driver/"))


from hypredriver import hypredriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from GPTune.gptune import * # import all

import re
import numpy as np
import time
import argparse
import pickle
from random import *
from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter
from GPTune.callhybrid import GPTuneHybrid
import math
import matplotlib.pyplot as plt
# from termcolor import colored

# import mpi4py
# from mpi4py import MPI



solver = 3 # Bommer AMG
max_setup_time = 100.
max_solve_time = 100.
coeffs_c = "-c 1 1 1 " # specify c-coefficients in format "-c 1 1 1 " 
coeffs_a = "-a 0 0 0 " # specify a-coefficients in format "-a 1 1 1 " leave as empty string for laplacian and Poisson problems
problem_name = "-laplacian " # "-difconv " for convection-diffusion problems to include the a coefficients

# define objective function
def objectives(point):

######################################### 
##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']	
#########################################
    print(point['x'])
    # task params 
    nx = point['nx']
    ny = point['ny']
    nz = point['nz']
    # tuning params / input params
    Px = 1 #point['Px']
    Py = 1 #point['Py']
    Nproc = 1 #point['Nproc']
    Pz = 1 #int(Nproc / (Px*Py))
    strong_threshold = 0.5 #point['strong_threshold']
    rounded = math.floor(min(point['x'],1-1e-12))
    trunc_factor =  point['x'] - rounded
    P_max_elmts = 6 # point['P_max_elmts']
    # coarsen_type = '6' # point['coarsen_type']
    coarsen_types = ['0', '1', '6', '8']
    coarsen_type = coarsen_types[rounded]
    relax_type = '0' #point['relax_type']
    smooth_type = '5' # point['smooth_type']
    smooth_num_levels = 0 # point['smooth_num_levels']
    interp_type = '0' #point['interp_type']
    agg_num_levels = 1 #point['agg_num_levels']

    # CoarsTypes = {0:"-cljp ", 1:"-ruge ", 2:"-ruge2b ", 3:"-ruge2b ", 4:"-ruge3c ", 6:"-falgout ", 8:"-pmis ", 10:"-hmis "}
    # CoarsType = CoarsTypes[coarsen_type]
    npernode =  1 # math.ceil(float(Nproc)/nodes)  
    nthreads = 1 #int(cores / npernode)
    
    # call Hypre 
    params = [(nx, ny, nz, coeffs_a, coeffs_c, problem_name, solver,
               Px, Py, Pz, strong_threshold, 
               trunc_factor, P_max_elmts, coarsen_type, relax_type, 
               smooth_type, smooth_num_levels, interp_type, agg_num_levels, nthreads, npernode)]
    runtime = hypredriver(params, niter=1, JOBID=-1)
    # print(params, colored(' hypre time: ','white','on_red'), runtime)
    print(params, ' hypre time: ', runtime)

    return runtime


def models(): # todo
    pass

def main(): 
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    # Parse command line arguments
    args = parse_args()

    nxmax = args.nxmax
    nymax = args.nymax
    nzmax = args.nzmax
    nprocmin_pernode = args.nprocmin_pernode
    ntask = args.ntask
    nrun = args.nrun
    npilot = args.npilot
    TUNER_NAME = args.optimization
    tla_II = args.tla_II

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores
    nprocmin = nodes*nprocmin_pernode 

    
    nxmin = 10
    nymin = 10
    nzmin = 10
    nx = Integer(nxmin, nxmax, transform="normalize", name="nx")
    ny = Integer(nymin, nymax, transform="normalize", name="ny")
    nz = Integer(nzmin, nzmax, transform="normalize", name="nz")
    # Px = Integer(1, nprocmax, transform="normalize", name="Px")
    # Py = Integer(1, nprocmax, transform="normalize", name="Py")
    # Nproc = Integer(nprocmin, nprocmax, transform="normalize", name="Nproc")
    # strong_threshold = Real(0, 1, transform="normalize", name="strong_threshold")
    trunc_factor =  Real(0, 1, transform="normalize", name="trunc_factor")
    # P_max_elmts = Integer(1, 12,  transform="normalize", name="P_max_elmts")
    coarsen_type = Categoricalnorm (['0', '1', '6', '8'], transform="onehot", name="coarsen_type")
    # relax_type = Categoricalnorm (['-1', '0', '6', '8', '16', '18'], transform="onehot", name="relax_type")
    # smooth_type = Categoricalnorm (['5', '6', '7', '8', '9'], transform="onehot", name="smooth_type")
    # smooth_num_levels = Integer(0, 5,  transform="normalize", name="smooth_num_levels")
    interp_type = Categoricalnorm (['0', '3', '4', '5', '6', '8', '12'], transform="onehot", name="interp_type")
    # agg_num_levels = Integer(0, 5,  transform="normalize", name="agg_num_levels")
    r = Real(float("-Inf"), float("Inf"), name="r")

    x =  Real(0, 4, transform="normalize", name="x")

    
    IS = Space([nx, ny, nz])
    # PS = Space([Px, Py, Nproc, strong_threshold, trunc_factor, P_max_elmts, coarsen_type, relax_type, smooth_type, smooth_num_levels, interp_type, agg_num_levels])
    # PS = Space([trunc_factor])
    PS = Space([x])
    OS = Space([r])
    
    # Question: how to set constraints
    # cst1 = f"Px * Py  <= Nproc"
    # cst2 = f"not(coarsen_type=='0' and P_max_elmts==10 and relax_type=='18' and smooth_type=='6' and smooth_num_levels==3 and interp_type=='8' and agg_num_levels==1)"
    # constraints = {"cst1": cst1,"cst2": cst2}
    constraints = {}
    constants={"nodes":nodes,"cores":cores}

    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants) 
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    # options['sample_class'] = 'SampleOpenTURNS'
    # Use the following two lines if you want to specify a certain random seed for surrogate modeling
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    # Use the following two lines if you want to specify a certain random seed for the search phase
    options['verbose'] = False
    options.validate(computer=computer)
    
    seed(1)
    if(ntask==1):
        giventask = [[nxmax,nymax,nzmax]]
    else:    
        giventask = [[randint(nxmin,nxmax),randint(nymin,nymax),randint(nzmin,nzmax)] for i in range(ntask)]
    # giventask = [[50, 60, 80], [60, 80, 100]]
    # # the following will use only task lists stored in the pickle file
    data = Data(problem)






    plot=1

    if plot==1:
        # fig = plt.figure(figsize=[12.8, 9.6])
        x = np.arange(0., 3.9999, 0.05)
        for tid in range(len(giventask)):
            fig = plt.figure(figsize=[12.8, 9.6])
            p = giventask[tid]
            t = p[0]
            I_orig=p
            kwargst = {IS[k].name: I_orig[k] for k in range(len(IS))}
            y=np.zeros([len(x),1])
            for i in range(len(x)):
                P_orig=[x[i]]
                kwargs = {PS[k].name: P_orig[k] for k in range(len(PS))}
                kwargs.update(kwargst)
                kwargs.update(constants)
                y[i]=objectives(kwargs)
            fontsize=40
            plt.rcParams.update({'font.size': 40})
            plt.plot(x, y, 'b',lw=2,label='true')

            plt.xlabel('x',fontsize=fontsize+2)
            plt.ylabel('y(x)',fontsize=fontsize+2)
            plt.title('t=%f'%t,fontsize=fontsize+2)
            print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())
            # legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            # legend = plt.legend(loc='upper right', shadow=False, fontsize=fontsize)
            # annot_min(x,y)
            # plt.show()
            plt.show(block=False)
            plt.pause(0.5)
            input("Press [enter] to continue.")
            fig.savefig('obj_nx_%f.pdf'%t)
            print(y)





    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nrun
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=npilot)
        print("stats: ", stats)
        
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if tla_II == 1:
            """ Call TLA for 2 new tasks """

            # the data object initialized to run transfer learning as a new autotuning run
            data = Data(problem)
            historydb=HistoryDB() #meta_dict=tuning_metadata)
            gt = GPTune(problem, computer=computer, data=data, options=options,historydb=historydb, driverabspath=os.path.abspath(__file__))

            # load source function evaluation data
            def LoadFunctionEvaluations(Tsrc):
                function_evaluations = [[] for i in range(len(Tsrc))]
                with open ("gptune.db/Hypre.json", "r") as f_in:
                    for func_eval in json.load(f_in)["func_eval"]:
                        task_parameter = [func_eval["task_parameter"]["nx"], func_eval["task_parameter"]["ny"], func_eval["task_parameter"]["nz"]]
                        if task_parameter in Tsrc:
                            function_evaluations[Tsrc.index(task_parameter)].append(func_eval)
                return function_evaluations

            newtask = [[20, 20, 20], [25, 25, 25]]
            (aprxopts, objval, stats) = gt.TLA_II(Tnew=newtask, Tsrc=giventask, source_function_evaluations=LoadFunctionEvaluations(giventask))
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])


    if(TUNER_NAME=='cgp'):
        from GPTune.callcgp import cGP
        NI = len(giventask)
        NS = nrun
        options['EXAMPLE_NAME_CGP']='Hypre'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


    if(TUNER_NAME=='opentuner'):
        NI = len(giventask)
        NS = nrun
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = len(giventask)
        NS = nrun
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='GPTuneHybrid'):
        NI = len(giventask)        
        options['n_budget_hybrid']=nrun
        options['n_pilot_hybrid']=max(nrun//2, 1)
        options['n_find_leaf_hybrid']=1
        # print(options)

        (data,stats) = GPTuneHybrid(T=giventask, tp=problem, computer=computer, options=options, run_id="GPTuneHybrid")
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))





def parse_args():
    parser = argparse.ArgumentParser()
    # Problem related arguments
    parser.add_argument('-nxmax', type=int, default=50, help='discretization size in dimension x')
    parser.add_argument('-nymax', type=int, default=50, help='discretization size in dimension y')
    parser.add_argument('-nzmax', type=int, default=50, help='discretization size in dimension y')
    # Machine related arguments
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    # Algorithm related arguments
    # parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune, GPTuneHybrid)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=-1, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=0, help='Number of runs per task')
    parser.add_argument('-tla_II', type=int, default=0, help='Whether perform TLA_II after MLA when optimization is GPTune')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
