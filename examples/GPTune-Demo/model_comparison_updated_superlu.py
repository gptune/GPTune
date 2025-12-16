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


"""
Example of invocation of this script:

mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

where:
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -perfmodel is whether a coarse performance model is used
    -optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""


################################################################################
import sys
import os
# import mpi4py
import logging
from pdbridge import *


logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from GPTune.gptune import * # import all
from mpl_toolkits.mplot3d import Axes3D


import argparse
# from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt

from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter



# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')

    args = parser.parse_args()

    return args

def objectives1(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1

    # print('test:',test)
    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f]

def objectives2(point):
    x1 = point["x1"]
    x2 = point["x2"]
    y = 4*(x1**2) + 4*(x2**2)
    return [y]

def objectives3(point):
    x1 = point["x1"]
    x2 = point["x2"]
    x3 = point["x3"]
    x4 = point["x4"]
    x5 = point["x5"]
    x6 = point["x6"]
    y = -1*(25*((x1-2)**2) + (x2-2)**2 + (x3-1)**2 + (x4-4)**2 + (x5-1)**2 + (x6-1)**2)
    return [y]


def predict_aug(modeler, gt, point,tid,objtype):   # point is the orginal space

    if(objtype==1):
        x =point['x']
        x = [x]
    elif(objtype==2):
        x1 =point['x1']
        x2 =point['x2']
        x = [x1,x2]  
    elif(objtype==3):
        x1 =point['x1']
        x2 =point['x2']          
        x3 =point['x3']
        x4 =point['x4']  
        x5 =point['x5']
        x6 =point['x6']  
        x = [x1,x2,x3,x4,x5,x6]  

    # x =point['x']
    xNorm = gt.problem.PS.transform([x])
    xi0 = gt.problem.PS.inverse_transform(np.array(xNorm, ndmin=2))
    xi=xi0[0]

    IOrig = gt.data.I[tid]

    # point0 = gt.data.D
    point2 = {gt.problem.IS[k].name: IOrig[k] for k in range(gt.problem.DI)}
    point  = {gt.problem.PS[k].name: xi[k] for k in range(gt.problem.DP)}
    # point.update(point0)
    point.update(point2)
    # print("point", point)

    xNorm = gt.problem.PS.transform(xi0)[0]
    if(gt.problem.models is not None):
        if(gt.problem.driverabspath is not None):
            modulename = Path(gt.problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(gt.problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module
        else:
            raise Exception('performance models require passing driverabspath to GPTune')
        # modeldata= self.problem.models(point)
        modeldata= module.models(point)
        xNorm = np.hstack((xNorm,modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
        # print(xNorm)
    (mu, var) = modeler[0].predict(xNorm, tid=tid)
    return (mu, var)


def model_runtime(model, obj_func, NS_input,objtype,optimizer,plotgp,lowrank=False, modelsparse=False):
    import matplotlib
    matplotlib.use('Agg')    
    import matplotlib.pyplot as plt
    global nodes
    global cores


    # Parse command line arguments
    args = parse_args()
    ntask = args.ntask
    nrun = NS_input
    tvalue = args.tvalue
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("\nmachine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    # parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    
    x = Real(0., 1., transform="normalize", name="x")
    x1 = Real(0., 1., transform="normalize", name="x1")
    x2 = Real(0., 1., transform="normalize", name="x2")
    x3 = Real(0., 1., transform="normalize", name="x3")
    x4 = Real(0., 1., transform="normalize", name="x4")
    x5 = Real(0., 1., transform="normalize", name="x5")
    x6 = Real(0., 1., transform="normalize", name="x6")    
    if(objtype==1):
        parameter_space = Space([x])    
    elif(objtype==2):
        parameter_space = Space([x1,x2])    
    elif(objtype==3):
        parameter_space = Space([x1,x2,x3,x4,x5,x6])    

    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    #constraints = {"cst1": "x >= 0. and x <= 1."}
    constraints = {}
    if(perfmodel==1):
        problem =  (input_space, parameter_space,output_space, obj_func, constraints, models)  # with performance model
    else:
        problem = TuningProblem(input_space, parameter_space,output_space, obj_func, constraints, None)  # no performance model

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    # options['objective_evaluation_parallelism'] = True
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 4
    # options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    options['model_optimzier'] = 'lbfgs'

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    # Use the following two lines if you want to specify a certain random seed for the random pilot sampling
    # options['sample_algo'] = 'MCS'
    options['sample_class'] = 'SampleOpenTURNS' #'SampleLHSMDU' 
    options['sample_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for surrogate modeling
    options['model_class'] = model #'Model_George_LCM'#'Model_George_LCM'  #'Model_LCM'
    options['model_kern'] = 'RBF' #'Matern32' #'RBF' #'Matern52'
    if(lowrank==True):
        options['model_lowrank'] = True
        options['model_hodlrleaf'] = 200
        options['model_hodlrtol'] = 1e-10
        options['model_hodlrtol_abs'] = 1e-20
        options['model_hodlr_sym'] = 0
        options['model_hodlr_knn'] = 0
        options['model_jitter'] = 0 # 1e-5 # 1e-3

    if(modelsparse==True):
        options['model_sparse'] = True
        options['model_cutoff'] = 0.1
        options['model_kern'] = 'WendlandC2'
    
    # Temporary hardcode 
    if(optimizer == "gradient"):
        options['model_mcmc'] = False
        options['model_grad'] = True
    elif (optimizer == "mcmc"):
        options['model_mcmc'] = True
        options['model_grad'] = False
        options['model_mcmc_sampler'] = 'MetropolisHastings' # 'Ensemble_emcee', 'MetropolisHastings'
        options['model_mcmc_nchain'] = 2
    elif (optimizer == "finite difference"):
        options['model_mcmc'] = False
        options['model_grad'] = False
    else:
        pass
        
    options['model_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for the search phase
    # options['search_class'] = 'SearchSciPy'
    options['search_random_seed'] = 0

    #options['search_class'] = 'SearchSciPy'
    #options['search_algo'] = 'l-bfgs-b'

    options['search_more_samples'] = 1
    options['search_af']='EI'
    options['search_pop_size']=100
    options['search_gen']=2
    # options['search_ucb_beta']=0.01


    options['verbose'] = True
    options['debug'] = False
    options.validate(computer=computer)

    # print(options)

    if ntask == 1:
        giventask = [[round(tvalue,1)]]
    elif ntask == 2:
        giventask = [[round(tvalue,1)],[round(tvalue*2.0,1)]]
    else:
        giventask = [[round(tvalue*float(i+1),1)] for i in range(ntask)]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']
    if(TUNER_NAME=='GPTune'):
        data = Data(problem)

        # gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__)) # run GPTune with database 
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__), historydb=False) ## Run GPTune without database

        # (data, modeler, stats) = gt.MLA(NS=NS_input, NS1= int(NS_input/2.0), NI=NI, Tgiven=giventask)
        (data, modeler, stats) = gt.MLA(NS=NS_input, NS1= int(NS_input - 1), NI=NI, Tgiven=giventask)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            # print("    Ps ", data.P[tid])
            # print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


        if objtype==1 and plotgp==True:
            # fig = plt.figure(figsize=[12.8, 9.6])
            for tid in range(len(data.I)):
                x = np.arange(0., 1., 0.01)
                fig = plt.figure(figsize=[12.8, 9.6])
                p = data.I[tid]
                t = p[0]
                I_orig=p
                kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}
                y=np.zeros([len(x),1])
                y_mean=np.zeros([len(x)])
                y_std=np.zeros([len(x)])
                for i in range(len(x)):
                    P_orig=[x[i]]
                    kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                    kwargs.update(kwargst)
                    y[i]=objectives1(kwargs)
                    if(TUNER_NAME=='GPTune'):
                        (y_mean[i],var) = predict_aug(modeler, gt, kwargs,tid,objtype)
                        y_std[i]=np.sqrt(var)
                        # print(y_mean[i],y_std[i],y[i])
                fontsize=40
                plt.rcParams.update({'font.size': 40})
                plt.plot(x, y, 'b',lw=2,label='true')

                plt.plot(x, y_mean, 'k', lw=3, zorder=9, label='prediction')
                plt.fill_between(x, y_mean - y_std, y_mean + y_std,alpha=0.2, color='k')
                # print(data.P[tid])
                plt.scatter(data.P[tid], data.O[tid], c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label='sample')

                plt.xlabel('x',fontsize=fontsize+2)
                plt.ylabel('y(t,x)',fontsize=fontsize+2)
                plt.title('t=%f'%t,fontsize=fontsize+2)
                print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())
                # legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
                legend = plt.legend(loc='upper right', shadow=False, fontsize=fontsize)
                # annot_min(x,y)
                # plt.show()
                plt.show(block=False)
                plt.pause(0.5)
                # input("Press [enter] to continue.")
                if(lowrank==True):
                    fig.savefig('obj_%s_N_%s_tol_%s.pdf'%(optimizer,int(NS_input - 1),options['model_hodlrtol']))
                elif(modelsparse==True):    
                    fig.savefig('obj_%s_N_%s_superlu.pdf'%(optimizer,int(NS_input - 1)))
                else:
                    fig.savefig('obj_%s_N_%s.pdf'%(optimizer,int(NS_input - 1)))
                    



        if objtype==2 and plotgp==True:
            # fig = plt.figure(figsize=[12.8, 9.6])
            for tid in range(len(data.I)):
                res = 80
                x1 = np.linspace(0., 1., res)
                x2 = np.linspace(0., 1., res)
                X1, X2 = np.meshgrid(x1, x2)

                Y_mean = np.zeros_like(X1)
                Y_std  = np.zeros_like(X1)
                Y_true  = np.zeros_like(X1)

                # kwargst from GPTune.data.I[tid]
                p = data.I[tid]
                t = p[0]
                kwargst = {input_space[k].name: p[k] for k in range(len(input_space))}

                # ====== 2. Evaluate GP ======
                for i in range(res):
                    for j in range(res):
                        P_orig = [x1[i], x2[j]]
                        kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                        kwargs.update(kwargst)
                        
                        Y_true[j, i] = objectives2(kwargs)[0]

                        y_m, var = predict_aug(modeler, gt, kwargs, tid, objtype)
                        # print(kwargs,y_m,var)
                        Y_mean[j, i] = y_m
                        Y_std[j, i]  = np.sqrt(var)
                # print(Y_mean)
                # Sampled points
                samples = np.array(data.P[tid], dtype=float)   # shape (m, 2)
                # print(samples)
                sample_vals = data.O[tid].reshape(-1)  # shape (m,)

                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                titles = ["True Function", "Predicted Mean", "Predicted Std"]

                # --- True function ---
                c0 = axes[0].contourf(X1, X2, Y_true, levels=50, cmap='viridis')
                fig.colorbar(c0, ax=axes[0])
                axes[0].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[0].set_title(titles[0])
                axes[0].set_xlabel("x1")
                axes[0].set_ylabel("x2")

                # --- GP mean ---
                c1 = axes[1].contourf(X1, X2, Y_mean, levels=50, cmap='viridis')
                fig.colorbar(c1, ax=axes[1])
                axes[1].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[1].set_title(titles[1])
                axes[1].set_xlabel("x1")
                axes[1].set_ylabel("x2")

                # --- GP std ---
                c2 = axes[2].contourf(X1, X2, Y_std, levels=50, cmap='magma')
                fig.colorbar(c2, ax=axes[2])
                axes[2].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[2].set_title(titles[2])
                axes[2].set_xlabel("x1")
                axes[2].set_ylabel("x2")

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.5)
                # input("Press [enter] to continue.")
                if(lowrank==True):
                    fig.savefig('obj_2D_%s_N_%s_tol_%s.pdf'%(optimizer,int(NS_input - 1),options['model_hodlrtol']))
                elif(modelsparse==True):    
                    fig.savefig('obj_2D_%s_N_%s_superlu.pdf'%(optimizer,int(NS_input - 1)))
                else:
                    fig.savefig('obj_2D_%s_N_%s.pdf'%(optimizer,int(NS_input - 1)))
                    

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from GPTune.callcgp import cGP
        options['EXAMPLE_NAME_CGP']='GPTune-Demo'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
    
    return stats


def objective_selection():
    objective = input("What Objective Function would you like to use (1, 2, or 3)")
    if ("1" in objective):
        objtype=1
        return objectives1, objtype
    elif ("2" in objective):
        objtype=2
        return objectives2, objtype
    elif ("3" in objective):
        objtype=3
        return objectives3, objtype
    else:
        raise Exception("Invalid objective selection")

    
    
    
import matplotlib.pyplot as plt

def plotting(objective, objtype):
    model_time_gpy = []
    model_time_per_likelihoodeval_gpy = []
    search_time_gpy = []
    model_iterations_gpy = []

    model_time_george_hodlr_gradient = []
    model_time_per_likelihoodeval_george_hodlr_gradient = []
    search_time_george_hodlr_gradient = []
    model_iterations_hodlr_gradient = []

    model_time_george_hodlr_finite_difference = []
    model_time_per_likelihoodeval_george_hodlr_finite_difference = []
    search_time_george_hodlr_finite_difference = []
    model_iterations_hodlr_finite_difference = []

    model_time_george_hodlr_mcmc = []
    model_time_per_likelihoodeval_george_hodlr_mcmc = []
    search_time_george_hodlr_mcmc = []
    model_iterations_hodlr_mcmc = []

    model_time_george_sparse_gradient = []
    model_time_per_likelihoodeval_george_sparse_gradient = []
    search_time_george_sparse_gradient = []
    model_iterations_sparse_gradient = []

    model_time_george_sparse_finite_difference = []
    model_time_per_likelihoodeval_george_sparse_finite_difference = []
    search_time_george_sparse_finite_difference = []
    model_iterations_sparse_finite_difference = []

    model_time_george_sparse_mcmc = []
    model_time_per_likelihoodeval_george_sparse_mcmc = []
    search_time_george_sparse_mcmc = []
    model_iterations_sparse_mcmc = []


    plotgp=True

    # NS = [201, 401, 801, 1601, 3201, 6401, 12801, 25601, 51201, 102401, 204801, 409601]
    # NS = [1601, 3201, 6401, 12801]
    # NS = [25601, 51201, 102401]
    # NS = [6401, 12801, 25601, 51201, 102401]
    NS = [6401]
    
    for elem in NS:

        sparse_stats_gradient = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, modelsparse=True, optimizer="gradient",plotgp=plotgp)
        model_time_george_sparse_gradient.append(sparse_stats_gradient.get("time_model"))
        model_time_per_likelihoodeval_george_sparse_gradient.append(sparse_stats_gradient.get("time_model_per_likelihoodeval"))
        search_time_george_sparse_gradient.append(sparse_stats_gradient.get("time_search"))
        model_iterations_sparse_gradient.extend(sparse_stats_gradient.get("modeling_iteration"))
        

        # sparse_stats_finite_difference = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, modelsparse=True, optimizer = "finite difference",plotgp=plotgp)
        # model_time_george_sparse_finite_difference.append(sparse_stats_finite_difference.get("time_model"))
        # model_time_per_likelihoodeval_george_sparse_finite_difference.append(sparse_stats_finite_difference.get("time_model_per_likelihoodeval"))
        # search_time_george_sparse_finite_difference.append(sparse_stats_finite_difference.get("time_search"))
        # model_iterations_sparse_finite_difference.extend(sparse_stats_finite_difference.get("modeling_iteration"))

        # sparse_stats_mcmc = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, modelsparse=True, optimizer="mcmc",plotgp=plotgp)
        # model_time_george_sparse_mcmc.append(sparse_stats_mcmc.get("time_model"))
        # model_time_per_likelihoodeval_george_sparse_mcmc.append(sparse_stats_mcmc.get("time_model_per_likelihoodeval"))
        # search_time_george_sparse_mcmc.append(sparse_stats_mcmc.get("time_search"))
        # model_iterations_sparse_mcmc.extend(sparse_stats_mcmc.get("modeling_iteration"))


        # hodlr_stats_gradient = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, lowrank=True, optimizer="gradient",plotgp=plotgp)
        # model_time_george_hodlr_gradient.append(hodlr_stats_gradient.get("time_model"))
        # model_time_per_likelihoodeval_george_hodlr_gradient.append(hodlr_stats_gradient.get("time_model_per_likelihoodeval"))
        # search_time_george_hodlr_gradient.append(hodlr_stats_gradient.get("time_search"))
        # model_iterations_hodlr_gradient.extend(hodlr_stats_gradient.get("modeling_iteration"))
        

        # hodlr_stats_finite_difference = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, lowrank=True, optimizer = "finite difference",plotgp=plotgp)
        # model_time_george_hodlr_finite_difference.append(hodlr_stats_finite_difference.get("time_model"))
        # model_time_per_likelihoodeval_george_hodlr_finite_difference.append(hodlr_stats_finite_difference.get("time_model_per_likelihoodeval"))
        # search_time_george_hodlr_finite_difference.append(hodlr_stats_finite_difference.get("time_search"))
        # model_iterations_hodlr_finite_difference.extend(hodlr_stats_finite_difference.get("modeling_iteration"))

        # hodlr_stats_mcmc = model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, lowrank=True, optimizer="mcmc",plotgp=plotgp)
        # model_time_george_hodlr_mcmc.append(hodlr_stats_mcmc.get("time_model"))
        # model_time_per_likelihoodeval_george_hodlr_mcmc.append(hodlr_stats_mcmc.get("time_model_per_likelihoodeval"))
        # search_time_george_hodlr_mcmc.append(hodlr_stats_mcmc.get("time_search"))
        # model_iterations_hodlr_mcmc.extend(hodlr_stats_mcmc.get("modeling_iteration"))


        # gpy_stats = model_runtime(model="Model_GPy_LCM", obj_func=objective, NS_input=elem, objtype=objtype, lowrank=False, optimizer = "Gpy_optimizer",plotgp=plotgp) 
        # model_time_gpy.append(gpy_stats.get("time_model"))
        # model_time_per_likelihoodeval_gpy.append(gpy_stats.get("time_model_per_likelihoodeval"))
        # search_time_gpy.append(gpy_stats.get("time_search"))
        # model_iterations_gpy.extend(gpy_stats.get("modeling_iteration"))
     
    


    # Model Time
    print("Time-Model George HODLR Gradient: ", model_time_george_hodlr_gradient)
    print("Time-Model George HODLR Finite Difference: ", model_time_george_hodlr_finite_difference)
    print("Time-Model George HODLR MCMC: ", model_time_george_hodlr_mcmc)
    print("Time-Model George Sparse Gradient: ", model_time_george_sparse_gradient)
    print("Time-Model George Sparse Finite Difference: ", model_time_george_sparse_finite_difference)
    print("Time-Model George Sparse MCMC: ", model_time_george_sparse_mcmc)    
    print("Time-Model GPy: ", model_time_gpy)

    # Search Time
    print("Time-Search George HODLR Gradient: ", search_time_george_hodlr_gradient)
    print("Time-Search George HODLR Finite Difference: ", search_time_george_hodlr_finite_difference)
    print("Time-Search George HODLR MCMC: ", search_time_george_hodlr_mcmc)
    print("Time-Search George Sparse Gradient: ", search_time_george_sparse_gradient)
    print("Time-Search George Sparse Finite Difference: ", search_time_george_sparse_finite_difference)
    print("Time-Search George Sparse MCMC: ", search_time_george_sparse_mcmc)    
    print("Time-Search GPy: ", search_time_gpy)

    # Inversion Time
    print("Inversion Time George HODLR Gradient: ", model_time_per_likelihoodeval_george_hodlr_gradient)
    print("Inversion Time George HODLR Finite Difference: ", model_time_per_likelihoodeval_george_hodlr_finite_difference)
    print("Inversion Time George HODLR MCMC: ", model_time_per_likelihoodeval_george_hodlr_mcmc)
    print("Inversion Time George Sparse Gradient: ", model_time_per_likelihoodeval_george_sparse_gradient)
    print("Inversion Time George Sparse Finite Difference: ", model_time_per_likelihoodeval_george_sparse_finite_difference)
    print("Inversion Time George Sparse MCMC: ", model_time_per_likelihoodeval_george_sparse_mcmc)    
    print("Inversion Time GPy: ", model_time_per_likelihoodeval_gpy)

    # Modeling Iterations
    print("Modeling Iterations George HODLR Gradient: ", model_iterations_hodlr_gradient)
    print("Modeling Iterations George HODLR Finite Difference: ", model_iterations_hodlr_finite_difference)
    print("Modeling Iterations George HODLR MCMC: ", model_iterations_hodlr_mcmc)
    print("Modeling Iterations George Sparse Gradient: ", model_iterations_sparse_gradient)
    print("Modeling Iterations George Sparse Finite Difference: ", model_iterations_sparse_finite_difference)
    print("Modeling Iterations George Sparse MCMC: ", model_iterations_sparse_mcmc)    
    print("Modeling Iterations GPy: ", model_iterations_gpy)

    fontsize=8
    plt.rcParams.update({'font.size': fontsize})
    figure, axis = plt.subplots(2,2)
    figure.suptitle("Optimizer Comparison 1D",fontsize=fontsize)

    # axis[0,0].loglog(NS, model_time_gpy, label="GPy", color="green", marker='o')
    axis[0,0].loglog(NS, model_time_george_hodlr_gradient, label="hodlr_grad", color="blue", marker='o')
    axis[0,0].loglog(NS, model_time_george_hodlr_finite_difference, label="hodlr_fd", color="red", marker='o')
    axis[0,0].loglog(NS, model_time_george_hodlr_mcmc, label="hodlr_mcmc", color="purple", marker='o')    
    axis[0,0].loglog(NS, model_time_george_sparse_gradient, label="sparse_grad", color="blue", marker='x')
    axis[0,0].loglog(NS, model_time_george_sparse_finite_difference, label="sparse_fd", color="red", marker='x')
    axis[0,0].loglog(NS, model_time_george_sparse_mcmc, label="sparse_mcmc", color="purple", marker='x')
    axis[0,0].legend(fontsize=fontsize-4)
    axis[0,0].set_title("Model Time",fontsize=fontsize)
    axis[0,0].set_xlabel("Sample Count",fontsize=fontsize)
    axis[0,0].set_ylabel("Time (sec)",fontsize=fontsize)

    # axis[0,1].loglog(NS, search_time_gpy, label="GPy", color="green", marker='o')
    axis[0,1].loglog(NS, search_time_george_hodlr_gradient, label="hodlr_grad", color="blue", marker='o')
    axis[0,1].loglog(NS, search_time_george_hodlr_finite_difference, label="hodlr_fd", color="red", marker='o')
    axis[0,1].loglog(NS, search_time_george_hodlr_mcmc, label="hodlr_mcmc", color="purple", marker='o')
    axis[0,1].loglog(NS, search_time_george_sparse_gradient, label="sparse_grad", color="blue", marker='x')
    axis[0,1].loglog(NS, search_time_george_sparse_finite_difference, label="sparse_fd", color="red", marker='x')
    axis[0,1].loglog(NS, search_time_george_sparse_mcmc, label="sparse_mcmc", color="purple", marker='x')    
    axis[0,1].legend(fontsize=fontsize-4)
    axis[0,1].set_title("Search Time",fontsize=fontsize)
    axis[0,1].set_xlabel("Sample Count",fontsize=fontsize)
    axis[0,1].set_ylabel("Time (sec)",fontsize=fontsize)

    # axis[1,0].loglog(NS, model_time_per_likelihoodeval_gpy, label="GPy", color="green", marker='o')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_hodlr_gradient, label="hodlr_grad", color="blue", marker='o')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_hodlr_finite_difference, label="hodlr_fd", color="red", marker='o')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_hodlr_mcmc, label="hodlr_mcmc", color="purple", marker='o')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_sparse_gradient, label="sparse_grad", color="blue", marker='x')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_sparse_finite_difference, label="sparse_fd", color="red", marker='x')
    axis[1,0].loglog(NS, model_time_per_likelihoodeval_george_sparse_mcmc, label="sparse_mcmc", color="purple", marker='x')    
    axis[1,0].legend(fontsize=fontsize-4)
    axis[1,0].set_title("Model Time Per Iteration",fontsize=fontsize)
    axis[1,0].set_xlabel("Number of Samples",fontsize=fontsize)
    axis[1,0].set_ylabel("Time (sec)",fontsize=fontsize)

    # axis[1,1].loglog(NS, model_iterations_gpy, label="GPy", color="green", marker='o')
    axis[1,1].loglog(NS, model_iterations_hodlr_gradient, label="hodlr_grad", color="blue", marker='o')
    axis[1,1].loglog(NS, model_iterations_hodlr_finite_difference, label="hodlr_fd", color="red", marker='o')
    axis[1,1].loglog(NS, model_iterations_hodlr_mcmc, label="hodlr_mcmc", color="purple", marker='o')
    axis[1,1].loglog(NS, model_iterations_sparse_gradient, label="sparse_grad", color="blue", marker='x')
    axis[1,1].loglog(NS, model_iterations_sparse_finite_difference, label="sparse_fd", color="red", marker='x')
    axis[1,1].loglog(NS, model_iterations_sparse_mcmc, label="sparse_mcmc", color="purple", marker='x')    
    axis[1,1].legend(fontsize=fontsize-4)
    axis[1,1].set_title("Model Iterations",fontsize=fontsize)
    axis[1,1].set_xlabel("Sample Count",fontsize=fontsize)
    axis[1,1].set_ylabel("Iterations",fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    plt.savefig('opttimizer_compare.pdf')

def main():
    objective, objtype = objective_selection()
    plotting(objective=objective, objtype=objtype)

    # superlu_terminate()


if __name__ == "__main__":
    main()

