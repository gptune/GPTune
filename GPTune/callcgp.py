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
import cGP.cGP_constrained_module as cgpc
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
from scipy.optimize import LinearConstraint
####################################################################################################



class cGP_constrained_gptune(cgpc.cGP_constrained):

    def __init__(self, t, tp, computer, options, historydb : HistoryDB = None):

        super().__init__(options)
        self.tp          = tp
        self.computer    = computer
        self.t           = t
        self.problem     = Problem(tp, driverabspath=None, models_update=None)
        self.NS          = self.N_PILOT + self.N_SEQUENTIAL
        self.count_runs  = 0
        self.timefun     = 0
        print(options)
        self.options = options
        self.bigval=options['BIGVAL_CGP']
        if (historydb is None):
            historydb = HistoryDB()
        self.historydb = historydb

    def get_bounds(self,restrict):
        #if restrict == 1:
        bds = []
        for p in self.tp.parameter_space.dimensions:
            if (isinstance(p, Real)):
                bds.append([p.bounds[0],p.bounds[1]])
            elif (isinstance(p, Integer)):
                bds.append([p.bounds[0],p.bounds[1]])
            elif (isinstance(p, Categorical)):
                raise Exception("cGP does not support Categorical parameter type")
            else:
                raise Exception("Unknown parameter type")
        bds = np.array(bds).astype(float)
        return bds

    def get_linear_constraint(self):
        linear_constraint = LinearConstraint([[1]*len(self.tp.parameter_space)], [-np.inf], [np.inf])
        return linear_constraint
        
    def f_truth(self,X):

        t1 = time.time_ns()
        t = self.t
        x =[]
        for n,p in zip(X[0],self.tp.parameter_space.dimensions):
            if (isinstance(p, Real)):
                x.append(n)
            elif (isinstance(p, Integer)):
                x.append(int(n))

        kwargs = {d.name: x[i] for (i, d) in enumerate(self.tp.parameter_space)}
        kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.tp.input_space)}
        kwargs2.update(kwargs)
        check_constraints = functools.partial(self.computer.evaluate_constraints, self.tp, inputs_only = False, kwargs = kwargs)
        cond = check_constraints(kwargs2)

        if (cond):
            #y = float(self.tp.objective(kwargs2)[0])

            transform_T = self.tp.input_space.transform([t])[0]
            transform_X = self.tp.parameter_space.transform(X)
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
            y = self.bigval

        print(t, x, y)
        
        t2 = time.time_ns()
        self.timefun=self.timefun+(t2-t1)/1e9
        sys.stdout.flush()

        self.count_runs += 1
        return np.array([y])

####################################################################################################

 

def cGP(T, tp : TuningProblem, computer : Computer, options: Options, run_id="cGP"):

    # Initialize
    X = []
    Y = []
    data = Data(tp)

    # Tune
    stats = {
        "time_total": 0,
        "time_fun": 0
    }

    timefun=0
    t1 = time.time_ns()
    print("Start cGP")
    for i in range(len(T)):
        cgp_runner=cGP_constrained_gptune(t=T[i], tp=tp, computer=computer, options=options)
        (xs,ys)=cgp_runner.run()
        tmp = xs.tolist()
        tmp1 = []
        for xx in tmp:
            tmp2 = []
            for val,p in zip(xx,tp.parameter_space.dimensions):
                if (isinstance(p, Real)):
                    val = val
                elif (isinstance(p, Integer)):
                    val = (int(val))
                tmp2.append(val)    
            tmp1.append(tmp2)
        # print(tmp1,ys)
        X.append(tmp1)
        Y.append(ys)


    print("End cGP")
    t2 = time.time_ns()
    stats['time_total'] = (t2-t1)/1e9
    stats['time_fun'] = timefun
    # Finalize

    data.I=T
    data.P=X
    data.O=Y
    # Finalize

    return (data, stats)
