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


import math
import numpy as np
import copy
import functools
import time
import mpi4py
from mpi4py import MPI

from autotune.problem import TuningProblem

from problem import Problem
from computer import Computer
from data import Data
from options import Options
from sample import *
from sample_LHSMDU import *
from sample_OpenTURNS import *
from model import *
from model_GPy import *
from model_cLCM import *
from model_PyDeepGP import *
from model_sghmc_dgp import *
from search import *
from search_PyGMO import *


def MLA2(self, N1, N2, **kwargs):

    """ Continuous Correlated Multi-task Learning Autotuning """

    print('\n\n\n------Starting MLA2 with %d warm-up runs and %d optimized runs'%(N1, N2))
    stats = {
        "time_total": 0,
        "time_sample_init": 0,
        "time_fun": 0,
        "time_search": 0,
        "time_model": 0
    }

    t0=time.time_ns()

    # initialize()

    kwargs.update(copy.deepcopy(self.options))

    sampler  =  eval(f'{kwargs["sample_class"]}()')
    modelers = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')] * self.problem.DO
    searcher =  eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')

    self.data = self.data.normalized()
    
    # sample()

    N3 = 50
    check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, kwargs = kwargs)
    t1=time.time_ns()
    (I, P) = sampler.sample_fusion(n_samples = N1, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
    (I2, P2) = sampler.sample_fusion(n_samples = N3, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
    t2=time.time_ns()
    stats["time_sample_init"] += (t2 - t1)/1e9
    t1=time.time_ns()
    O = self.computer.evaluate_objective(self.problem, I, P, None, options = kwargs) 
    O2 = self.computer.evaluate_objective(self.problem, I2, P2, None, options = kwargs) 
    t2=time.time_ns()
    stats["time_fun"] += (t2 - t1)/1e9
    newdata = Data(problem = self.problem, I = I, P = P, O = O)
    testdata = Data(problem = self.problem, I = I2, P = P2, O = O2)
    self.data.fusion(newdata)

    for optiter in range(N2):

        print("MLA2 iteration: ", optiter)

        # model()

        if (optiter % kwargs['model_update_no_train_iters'] == 0):
            for o in range(self.problem.DO):
                tmpdata = copy.deepcopy(self.data)
                tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]
                t1=time.time_ns()
                modelers[o].train(tmpdata, **kwargs)
                t2=time.time_ns()
                stats["time_model"] += (t2 - t1)/1e9
        else:
            for o in range(self.problem.DO):
                tmpdata = copy.deepcopy(newdata)
                tmpdata.O = [copy.deepcopy(newdata.O[i][:,o].reshape((-1,1))) for i in range(len(newdata.I))]
                t1=time.time_ns()
                modelers[o].update(tmpdata, do_train=False, **kwargs)
                t2=time.time_ns()
                stats["time_model"] += (t2 - t1)/1e9
        Xts, Yts = testdata.IPO2XY()
        for o in range(self.problem.DO):
            mu, var = modelers[o].M.predict(Xts)
            print('RMSE', np.linalg.norm(mu - Yts))
            print('VAR', np.linalg.norm(var))

        # search()

        t1=time.time_ns()
        res = searcher.search(data = self.data, models = modelers, tid = None, sampler = sampler, **kwargs)[1]
        t2=time.time_ns()
        stats["time_search"] += (t2 - t1)/1e9

        newdata = Data(problem = self.problem)
        newdata.I = [x[0][:self.problem.DI] for x in res]
        newdata.P = [np.array(x[0][self.problem.DI:], ndmin=2) for x in res]
        t1=time.time_ns()
        newdata.O = self.computer.evaluate_objective(problem = self.problem, I = newdata.I, P = newdata.P, D = newdata.D, options = kwargs)
        t2 = time.time_ns()
        stats["time_fun"] += (t2 - t1)/1e9

        self.data.fusion(newdata)

    # finalize()

    self.data = self.data.originalized()

    t3 = time.time_ns()
    stats["time_total"] += (t3 - t0)/1e9

    return (copy.deepcopy(self.data), modelers, stats)

