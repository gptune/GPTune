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


import numpy as np
from autotune.problem import TuningProblem
from .problem import Problem
from .options import Options
from .computer import Computer
from .database import HistoryDB
from typing import Collection
import skopt.space
from skopt.space import *
from data import *
import math
import argparse
import functools
import time
import sys
import hybridMinimization.hybridMinimization as gptunehybrid
####################################################################################################



class gptunehybrid_worker(object):

    def __init__(self, t, tp, computer, options, historydb : HistoryDB = None):

        self.tp          = tp
        self.computer    = computer
        self.t           = t
        self.problem     = Problem(tp, driverabspath=None, models_update=None)
        self.NS          = options['n_budget_hybrid']
        self.count_runs  = 0
        self.timefun     = 0
        self.bigval=options['bigval_hybrid']
        # print(options)
        self.options = options
        if (historydb is None):
            historydb = HistoryDB()
        self.historydb = historydb

        categorical_list=[]
        continuous_list=[]
        for p in self.tp.parameter_space.dimensions:
            if (isinstance(p, Real)):
                continuous_list.append([p.bounds[0],p.bounds[1]])
            elif (isinstance(p, Integer)):
                continuous_list.append([p.bounds[0],p.bounds[1]])
            elif (isinstance(p, Categorical)):
                categorical_list.append(list(range(len(list(p.bounds)))))
            else:
                raise Exception("Unknown parameter type")
        # print(categorical_list)
        # print(continuous_list)
        self.categorical_list = categorical_list
        self.continuous_list = continuous_list


    #########  The input of f_truth assumes categorical variables (with integer choices) followed by continous variables
    #########  The following function converts it to the input of GPTune objective functions
    def hybridx_to_gptunex(self,X):
        kwargs={}
        idx_cont=0
        idx_cat=0
        x =[]
        for n,p in zip(self.tp.parameter_space.dimension_names,self.tp.parameter_space.dimensions):
            if (isinstance(p, Real)):
                val = X[len(self.categorical_list)+idx_cont]
                kwargs[n]=val
                idx_cont=idx_cont+1
            elif (isinstance(p, Integer)):
                val = X[len(self.categorical_list)+idx_cont]
                kwargs[n]=int(val)
                idx_cont=idx_cont+1
            elif (isinstance(p, Categorical)):
                val = int(X[idx_cat])
                kwargs[n]=list(p.bounds)[val]
                idx_cat=idx_cat+1
            else:
                raise Exception("Unknown parameter type")
            x.append(kwargs[n])
        return (x,kwargs)



    #########  The following function converts input of GPTune to the hybrid code, note that gptunex_to_hybridx(hybridx_to_gptunex(X)) may not be equal to X due to integer rounding
    def gptunex_to_hybridx(self,x):
        kwargs={}
        idx_cont=0
        idx_cat=0
        X =['tmp']*len(x)
        nn=0
        for n,p in zip(self.tp.parameter_space.dimension_names,self.tp.parameter_space.dimensions):
            if (isinstance(p, Real)):
                X[len(self.categorical_list)+idx_cont]=x[nn]
                idx_cont=idx_cont+1
            elif (isinstance(p, Integer)):
                X[len(self.categorical_list)+idx_cont]=int(x[nn])
                idx_cont=idx_cont+1
            elif (isinstance(p, Categorical)):
                idx=list(p.bounds).index(x[nn])
                X[idx_cat]=idx
                idx_cat=idx_cat+1
            else:
                raise Exception("Unknown parameter type")
            nn=nn+1
        return X


    def blk_constraint(self,X):

        t = self.t

        (x,kwargs) = self.hybridx_to_gptunex(X)

        kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.tp.input_space)}
        kwargs2.update(kwargs)
        check_constraints = functools.partial(self.computer.evaluate_constraints, self.tp, inputs_only = False, kwargs = kwargs)
        cond = check_constraints(kwargs2)
        return cond


    def f_truth(self,X):

        t1 = time.time_ns()
        t = self.t

        (x,kwargs) = self.hybridx_to_gptunex(X)
        # Xtmp=self.gptunex_to_hybridx(x)

        kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.tp.input_space)}
        kwargs2.update(kwargs)
        check_constraints = functools.partial(self.computer.evaluate_constraints, self.tp, inputs_only = False, kwargs = kwargs)
        cond = check_constraints(kwargs2)

        if (cond):
            #y = float(self.tp.objective(kwargs2)[0])

            transform_T = self.tp.input_space.transform([t])[0]
            transform_X = self.tp.parameter_space.transform([x])
            if(self.options['RCI_mode']==False):
                result = self.computer.evaluate_objective_onetask(
                        problem = self.problem,
                        i_am_manager = True,
                        T2 = transform_T,
                        P2 = transform_X,
                        D2 = {},
                        history_db = self.historydb,
                        options = self.options
                        )
                y = result[0][0]
                #print ("evaluate_objective_onetask result: ", result)
            else:
                tmp = np.empty( shape=(len(transform_X), self.problem.DO))
                tmp[:] = np.NaN
                modeling = "SLA_GP"

                self.historydb.store_func_eval(problem = self.problem,\
                        task_parameter = transform_T, \
                        tuning_parameter = transform_X,\
                        evaluation_result = tmp,\
                        evaluation_detail = tmp,\
                        source = "RCI_measure",\
                        modeling = modeling,\
                        model_class = self.options["model_class"])
                print('RCI: GPTune returns\n')
                exit()

        else:
            y = self.bigval

        print(t, x, y)

        t2 = time.time_ns()
        self.timefun=self.timefun+(t2-t1)/1e9
        sys.stdout.flush()

        self.count_runs += 1
        return np.array([y])

####################################################################################################


def GPTuneHybrid(T, tp : TuningProblem, computer : Computer, options: Options, run_id="GPTuneHybrid"):
    # Initialize
    X = []
    Y = []
    data = Data(tp)
    time_fun=0

    # Tune
    stats = {
        "time_total": 0,
        "time_fun": 0,
        "count_runs": 0
    }

    t1 = time.time_ns()
    print("Start GPTuneHybrid")
    for i in range(len(T)):

        worker=gptunehybrid_worker(t=T[i], tp=tp, computer=computer, options=options)
        # (xs,ys)=cgp_runner.run()

        np.random.seed(options['random_seed_hybrid'])

        X0=[]
        Y0=[]
        ii=0

        data = Data(worker.problem)
        worker.historydb.load_history_func_eval(data, worker.problem, [T[i]], function_evaluations= None, source_function_evaluations=None, options=None)
        if(data.P is not None):
            ii = len(data.P[0])
            for iii in range(ii):
                Xtmp=worker.gptunex_to_hybridx(data.P[0][iii])
                X0.append(Xtmp)
                Y0.append(data.O[0][iii,:])

            if(options['RCI_mode']==True):
                np.random.seed(options['random_seed_hybrid']+ii+1) # this makes sure that the random samples are not the same

        while ii<int(options['n_pilot_hybrid']):
            x0_cat = [np.random.choice(i,size=1)[0] for i in worker.categorical_list]
            x0_con = [np.random.uniform(low=j[0],high=j[1],size=1)[0] for j in worker.continuous_list]
            x0 = x0_cat+x0_con

            (x,kwargs) = worker.hybridx_to_gptunex(x0)
            t = worker.t
            kwargs2 = {d.name: t[i] for (i, d) in enumerate(tp.input_space)}
            kwargs2.update(kwargs)
            check_constraints = functools.partial(computer.evaluate_constraints, tp, inputs_only = False, kwargs = kwargs)
            cond = check_constraints(kwargs2)
            if (cond):
                X0.append(x0)
                y0 = worker.f_truth(x0)
                Y0.append(y0)
                ii=ii+1


        X0=np.asarray(X0)
        Y0=np.asarray(Y0)
        # print(X0)
        # print(Y0)

        h1_y,h1_x,h1_root,h1_model,h1_model_history = gptunehybrid.hybridMinimization(fn=worker.f_truth, blkcst=worker.blk_constraint, \
                                                selection_criterion = options['selection_criterion_hybrid'],fix_model = -1,\
                                                categorical_list=worker.categorical_list,\
                                                categorical_trained_model=None,\
                                                policy_list=[options['policy_hybrid']]*len(worker.categorical_list),update_list=[options['policy_hybrid']]*len(worker.categorical_list),exploration_probability=options['exploration_probability_hybrid'],\
                                                continuous_list=worker.continuous_list,\
                                                continuous_trained_model=None,\
                                                observed_X = X0,\
                                                observed_Y = Y0,\
                                                n_find_tree=int(options['n_budget_hybrid']/options['n_find_leaf_hybrid']),\
                                                n_find_leaf=options['n_find_leaf_hybrid'],\
                                                node_optimize=options['acquisition_GP_hybrid'],\
                                                random_seed=options['random_seed_hybrid'],\
                                                N_initialize_leaf=0)

        # pkl_dict = {'categorical_model':h1_root,'continuous_model':h1_model,'train_X':h1_x,'train_Y':h1_y,'model_history':h1_model_history,'continuous_list':worker.continuous_list,'categorical_list':worker.categorical_list}
        # print(pkl_dict)

        tmp1 = []
        for ii in range(h1_x.shape[0]):
            tmp = h1_x[ii].tolist()
            (x,x_dict) = worker.hybridx_to_gptunex(tmp)
            tmp1.append(x)
        X.append(tmp1)
        Y.append(h1_y)

        time_fun = time_fun + worker.timefun

    print("End GPTuneHybrid")
    t2 = time.time_ns()
    stats['time_total'] = (t2-t1)/1e9
    stats['time_fun'] = time_fun
    stats['count_runs'] = worker.count_runs
    # Finalize

    data.I=T
    data.P=X
    data.O=Y
    # Finalize

    return (data, stats)
