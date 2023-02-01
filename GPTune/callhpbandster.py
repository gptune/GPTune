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
import ConfigSpace
import hpbandster
import hpbandster.core.worker
import hpbandster.core.nameserver
import hpbandster.optimizers

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
####################################################################################################

class HpBandSterWorker(hpbandster.core.worker.Worker):

    def __init__(self, t, NS, tp, computer, historydb : HistoryDB = None, options : Options = None, niter=1, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.myworker_id = kwargs['id']
        self.tp          = tp
        self.computer    = computer
        self.problem     = Problem(tp, driverabspath=None, models_update=None)
        self.t           = t
        self.NS          = NS
        self.niter       = niter
        self.count_runs  = 0
        self.timefun     = 0
        if (options is None):
            options = Options()
        self.options = options
        if (historydb is None):
            historydb = HistoryDB()
        self.historydb = historydb

    def get_configspace(self):

        class MyConstrainedConfigurationSpace(ConfigSpace.ConfigurationSpace):

            def __init__(self, tp, computer, t):

                super(MyConstrainedConfigurationSpace, self).__init__()
                self.t = t
                self.tp = tp
                self.computer = computer

            def sample_configuration(self, size=1):

                cond = False
                cpt = 0
                while (not cond):

                    cpt += 1
                    if (size == 1):
                        accepted_configurations = [super(MyConstrainedConfigurationSpace, self).sample_configuration(size=size)]
                    else:
                        accepted_configurations = super(MyConstrainedConfigurationSpace, self).sample_configuration(size=size)
                    for config in accepted_configurations:
                        t = self.t
                        print(config,type(config))
                        x = [config[p] for p in self.tp.parameter_space.dimension_names]
                        kwargs = {d.name: x[i] for (i, d) in enumerate(self.tp.parameter_space)}
                        kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.tp.input_space)}
                        kwargs2.update(kwargs)
                        check_constraints = functools.partial(self.computer.evaluate_constraints, self.tp, inputs_only = False, kwargs = kwargs)
                        cond = check_constraints(kwargs2)
                        if (not cond):
                            break

                if (size == 1):
                    return accepted_configurations[0]
                else:
                    return accepted_configurations

        #config_space = ConfigSpace.ConfigurationSpace()
        config_space = MyConstrainedConfigurationSpace(self.tp, self.computer, self.t)

        for n,p in zip(self.tp.parameter_space.dimension_names,self.tp.parameter_space.dimensions):
            if (isinstance(p, Real)):
                config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(n, lower = p.bounds[0], upper = p.bounds[1]))
            elif (isinstance(p, Integer)):
                config_space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(n, lower = p.bounds[0], upper = p.bounds[1]))
            elif (isinstance(p, Categorical)):
                config_space.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(n, choices = list(p.bounds)))
            else:
                raise Exception("Unknown parameter type")
        return(config_space)

    def compute(self, config, budget, **kwargs):

        if (self.count_runs >= self.NS):

            y = float("Inf")
            state = 'OVERTIME'

            print(self.t, state)
            sys.stdout.flush()

        else:
            t1 = time.time_ns()
            t = self.t
            x = [config[p] for p in self.tp.parameter_space.dimension_names]
            kwargs = {d.name: x[i] for (i, d) in enumerate(self.tp.parameter_space)}
            kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.tp.input_space)}
            kwargs2.update(kwargs)
            kwargs2['budget'] = budget
            check_constraints = functools.partial(self.computer.evaluate_constraints, self.tp, inputs_only = False, kwargs = kwargs)
            cond = check_constraints(kwargs2)

            if (cond):
                #y = float(self.tp.objective(kwargs2)[0])

                transform_T = self.tp.input_space.transform([t])[0]
                transform_X = self.tp.parameter_space.transform([x])
                result = self.computer.evaluate_objective_onetask(
                        problem = self.problem,
                        i_am_manager = True,
                        T2 = transform_T,
                        P2 = transform_X,
                        D2 = {},
                        history_db = self.historydb,
                        options = self.options
                        )
                y = float(result[0][0])
                state = 'OK'
            else:
                y = float("Inf")
                state = 'ERROR'
            print("T X Y state")
            print(t, x, y, state)
            t2 = time.time_ns()
            self.timefun=self.timefun+(t2-t1)/1e9
            sys.stdout.flush()

        self.count_runs += 1

        return({
                   'loss': y,
                   'info': {"state":state}
               })

####################################################################################################

def HpBandSter(T, NS, tp : TuningProblem, computer : Computer, run_id="HpBandSter", niter=1):

    # Initialize
    min_budget   = 1. # Minimum budget used during the optimization.
    max_budget   = 1. # Maximum budget used during the optimization.
    n_iterations = NS # Number of iterations performed by the optimizer
    n_workers    = 1  # Number of workers to run in parallel.

    X = []
    Y = []
    # Xopt = []
    # Yopt = []
    data = Data(tp)

    server = hpbandster.core.nameserver.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    server.start()

    # Tune
    stats = {
        "time_total": 0,
        "time_fun": 0
    }

    timefun=0
    t1 = time.time_ns()
    print("Start HpBandSter")
    for i in range(len(T)):

        workers=[]
        for j in range(n_workers):
            w = HpBandSterWorker(t=T[i], NS=NS, tp=tp, computer=computer, niter=niter, run_id=run_id, nameserver='127.0.0.1', id=j)
            w.run(background=True)
            workers.append(w)

        bohb = hpbandster.optimizers.BOHB(configspace=workers[0].get_configspace(), run_id=run_id, nameserver='127.0.0.1', min_budget=min_budget, max_budget=max_budget)
        res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)

        config_mapping = res.get_id2config_mapping()
        # incumbent = res.get_incumbent_id()

        xs = [[config_mapping[idx]['config'][p] for p in tp.parameter_space.dimension_names] for idx in config_mapping.keys()]
        ys = [[v['loss'] for k,v in res[idx].results.items()] for idx in config_mapping.keys()]
        # xopt = np.array([res.get_id2config_mapping()[incumbent]['config'][p] for p in tp.parameter_space.dimension_names])
        # yopt = min([v['loss'] for k,v in res[incumbent].results.items()])
        X.append(xs)
        tmp = np.array(ys).reshape((len(ys), 1))
        Y.append(tmp)
        # Xopt.append(xopt)
        # Yopt.append(yopt)
        timefun=timefun+workers[0].timefun
        bohb.shutdown(shutdown_workers=True)

    print("End HpBandSter")
    t2 = time.time_ns()
    stats['time_total'] = (t2-t1)/1e9
    stats['time_fun'] = timefun
    # Finalize

    server.shutdown()

    data.I=T
    data.P=X
    data.O=Y
    # Finalize

    return (data, stats)

def HpBandSter_bandit(T, NS, tp : TuningProblem, computer : Computer, options: Options = None, run_id="HpBandSter_bandit", niter=1):
   # Initialize
    min_budget   = options['budget_min'] # Minimum budget used during the optimization.
    max_budget   = options['budget_max'] # Maximum budget used during the optimization.
    budget_base  = options['budget_base']
    n_iterations = NS # Number of iterations performed by the optimizer
    n_workers    = 1  # Number of workers to run in parallel.
    
    X = []
    Y = []
    # Xopt = []
    # Yopt = []
    data = Data(tp)

    server = hpbandster.core.nameserver.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    server.start()

    # Tune
    stats = {
        "time_total": 0,
        "time_fun": 0
    }

    timefun=0
    t1 = time.time_ns()
    for i in range(len(T)):

        workers=[]
        for j in range(n_workers):
            w = HpBandSterWorker(t=T[i], NS=NS, tp=tp, computer=computer, niter=niter, run_id=run_id, nameserver='127.0.0.1', id=j)
            w.run(background=True)
            workers.append(w)
            
        bohb = hpbandster.optimizers.BOHB(configspace=workers[0].get_configspace(), run_id=run_id, nameserver='127.0.0.1', min_budget=min_budget, max_budget=max_budget, eta=budget_base)
        res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)

        config_mapping = res.get_id2config_mapping()

        xs = [[config_mapping[idx]['config'][p] for p in tp.parameter_space.dimension_names] for idx in config_mapping.keys()]
        ys = [[(k, v['loss']) for k,v in res[idx].results.items()] for idx in config_mapping.keys()]
        
        X.append(xs)
        tmp = np.array(ys).reshape((len(ys), 1))
        Y.append(tmp)
        timefun=timefun+workers[0].timefun
        bohb.shutdown(shutdown_workers=True)

    t2 = time.time_ns()
    stats['time_total'] = (t2-t1)/1e9	
    stats['time_fun'] = timefun
    # Finalize

    server.shutdown()

    data.I=T
    data.P=X
    data.O=Y
    # Finalize

    return (data, stats)
