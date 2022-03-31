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
import sys
import os
# import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all


import mpi4py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from callopentuner import OpenTuner
from callhpbandster import HpBandSter, HpBandSter_bandit
import scipy


# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]

# bandit structure
bmin = 1
bmax = 4
eta = 4

def objectives(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    t = point['t']
    x = point['x']
    if 'budget' in point:
        bgt = point['budget']
    else:
        bgt = bmax

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
    def perturb(bgt):
        perturb_magnitude = 0.1
        k1 = -perturb_magnitude/bmax
        # return np.cos(c)*(-np.log10(bgt))*0.1
        assert k1*bmax + perturb_magnitude == 0
        return np.cos(c) * (k1*bgt + perturb_magnitude)

    out = [f*(1+perturb(bgt))]
    print(f"One demo run, x = {x:.4f}, t = {t:.4f}, budget = {bgt:.4f}, perturb = {perturb(bgt):.4f}, out = {out[0]:.4f}")
    return out


""" Plot the objective function for t=1,2,3,4,5,6 """
def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="offset points",arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(210,5), **kw)


def main():
    import matplotlib.pyplot as plt
    args = parse_args()
    ntask = args.ntask
    nruns = args.nruns
    TUNER_NAME = args.optimization
    Nloop = args.Nloop
    plot = args.plot
    expid = args.expid
    restart = args.restart

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    # problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, models)  # with performance model
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)  # no performance model

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options['model_restarts'] = restart

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
    #options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['verbose'] = False
    # options['sample_algo'] = 'MCS'
    # options['sample_class'] = 'SampleLHSMDU'
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
    Btotal = int(np.dot(np.array(NSs_all), np.array(budget_all))/options['budget_max']) # total number of evaluations at highest budget -- used for single-fidelity tuners
    print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)
    print("total number of samples: ", Ntotal)
    print("total number of evaluations at highest budget: ", Btotal)
    print(f"Sampler: {options['sample_class']}, {options['sample_algo']}")
    print()

    data = Data(problem)
    # giventask = [[1.0], [5.0], [10.0]]
    # giventask = [[1.0], [1.2], [1.3]]
    # giventask = [[1.0]]
    # t_end = args.t_end
    giventask = [[i] for i in np.arange(1, ntask/2+1, 0.5).tolist()]
    # giventask = [[i] for i in np.arange(1, 1.5, 0.05).tolist()]
    # giventask = [[1.0], [1.05], [1.1]]
    NI=len(giventask)
    assert NI == ntask # make sure number of tasks match

    np.set_printoptions(suppress=False, precision=3)
    if(TUNER_NAME=='GPTuneBand'):
        NS = Nloop
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
        (data, stats, data_hist)=gt.MB_LCM(NS = Nloop, Igiven = giventask)
        print("Tuner: ", TUNER_NAME)
        print("Sampler class: ", options['sample_class'])
        print("Model class: ", options['model_class'])
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   t = {data.I[tid][0]:.2f}")
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
        if args.nruns > 0:
            NS = args.nruns
            print("In GPTune, using the given number of nruns ", NS)
        NS1 = max(NS//2, 1)
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
        """ Building MLA with the given list of tasks """
        (data, modeler, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS1)
        print("stats: ", stats)
        print("model class: ", options['model_class'])
        print("Model restart: ", restart)

        """ Print all input and parameter samples """
        sum_Oopt = 0.
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], f'Oopt  {min(data.O[tid])[0]:.3f}', 'nth ', np.argmin(data.O[tid]))
            sum_Oopt += min(data.O[tid])[0]
        # print("sum of all optimal objectives", sum_Oopt)

    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid][:NS])], 'Oopt ', min(data.O[tid][:NS])[0], 'nth ', np.argmin(data.O[tid][:NS]))

    # single fidelity version of hpbandster
    if(TUNER_NAME=='TPE'):
        NS = Btotal
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    # multi-fidelity version
    if(TUNER_NAME=='hpbandster'):
        NS = Ntotal
        (data,stats)=HpBandSter_bandit(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
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

    if plot==1:
        x = np.arange(0., 1., 0.0001)
        ymean_set = [] # stores predicted function values
        ytrue_set = []
        for tid in range(len(data.I)):
            p = data.I[tid]
            t = p[0]
            fig = plt.figure(figsize=[12.8, 9.6])
            I_orig=p
            kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}
            y=np.zeros([len(x),1])
            y_mean=np.zeros([len(x)])
            y_std=np.zeros([len(x)])
            for i in range(len(x)):
                P_orig=[x[i]]
                kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                kwargs.update(kwargst)
                y[i]=objectives(kwargs)
                if(TUNER_NAME=='GPTune'):
                    (y_mean[i],var) = predict_aug(modeler, gt, kwargs,tid)
                    y_std[i]=np.sqrt(var)
                    # print(y_mean[i],y_std[i],y[i])
            fontsize=40
            plt.rcParams.update({'font.size': 40})
            plt.plot(x, y, 'b',lw=2,label='true')

            plt.plot(x, y_mean, 'k', lw=3, zorder=9, label='prediction')
            plt.fill_between(x, y_mean - y_std, y_mean + y_std,alpha=0.2, color='k')
            plt.ylim(0, 2)
            # print(data.P[tid])
            plt.scatter(data.P[tid], data.O[tid], c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label='sample')

            plt.xlabel('x',fontsize=fontsize+2)
            plt.ylabel('y(t,x)',fontsize=fontsize+2)
            plt.title('t=%f'%t,fontsize=fontsize+2)
            print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())
            # legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            legend = plt.legend(loc='upper right', shadow=False, fontsize=fontsize)
            annot_min(x,y)
            # plt.show()
            plt.show(block=False)
            plt.pause(0.5)
            # input("Press [enter] to continue.")
            # fig.savefig('obj_t_%f.eps'%t)
            fig.savefig(f'obj_ntask{NI}_{expid}_tid_{tid}_t_{t:.1f}.pdf')
            ymean_set.append(y_mean)
            ytrue_set.append(y)
        # show the distance among surrogate functions
        R = np.zeros((NI, NI)) # Pearson sample correlation matrix of learned surrogates
        R_true = np.zeros((NI, NI))# Pearson sample correlation of true functions
        for i in range(NI):
            for ip in range(i, NI):
                ymean_i = ymean_set[i]
                ymean_ip = ymean_set[ip]
                ytrue_i = np.array((ytrue_set[i]).reshape((1, -1)))[0]
                ytrue_ip = np.array((ytrue_set[ip]).reshape((1, -1)))[0]
                # find the Pearson sample correlation coefficient
                R[i, ip], _ = scipy.stats.pearsonr(ymean_i, ymean_ip)
                R_true[i, ip], _ = scipy.stats.pearsonr(ytrue_i, ytrue_ip)
        print("The correlation matrix among surrogate functions is: \n", R)
        print("The correlation matrix among true functions is: \n", R_true)
        new_Rtrue = R_true[np.triu_indices(R_true.shape[0], 1)]
        new_R = R[np.triu_indices(R.shape[0], 1)]
        print("The mean absolute error is: \n", np.mean(abs(new_Rtrue - new_R)))
        print("The mean relative error is: \n", np.mean( abs(new_Rtrue - new_R)/ abs(new_R) ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    # parser.add_argument('-t_end', type=float, default=2.0, help='end of task value')
    parser.add_argument('-nruns', type=int, default=-1, help='total application runs')
    parser.add_argument('-plot', type=int, default=0, help='Whether to plot the objective function')
    # parser.add_argument('-LCMmodel', type=str, default='LCM', help='choose from LCM models: LCM or GPy_LCM')
    parser.add_argument('-Nloop', type=int, default=1, help='Number of outer loops in multi-armed bandit per task')
    parser.add_argument('-expid', type=str,default='-1', help='experiment id')
    parser.add_argument('-restart', type=int, default=1, help='number of model restart')
    # parser.add_argument('-sample_class', type=str,default='SampleOpenTURNS',help='Supported sample classes: SampleLHSMDU, SampleOpenTURNS')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
