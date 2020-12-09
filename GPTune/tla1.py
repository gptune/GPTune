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


def TLA1(self, Tnew, NS):

    print('\n\n\n------Starting TLA1 for task: ',Tnew)

    stats = {
        "time_total": 0,
        "time_fun": 0
    }
    time_fun=0

    t3=time.time_ns()
    # Initialization
    kwargs = copy.deepcopy(self.options)
    ntso = len(self.data.I)
    ntsn = len(Tnew)

    if(self.data.O[0].shape[1]>1):
        raise Exception("TLA1 only works for single-objective tuning")

    PSopt =[]
    for i in range(ntso):
        PSopt.append(self.data.P[i][np.argmin(self.data.O[i])])
    # YSopt = np.array([[self.data.O[k].min()] for k in range(ntso)])
    MSopt = []



    # convert the task spaces to the normalized spaces
    INorms=[]
    for t in self.data.I:
        INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
        INorms.append(INorm.reshape((-1, self.problem.DI)))
    INorms = np.vstack([INorms[i] for i in range(ntso)]).reshape((ntso,self.problem.DI))

    tmp=[]
    for t in Tnew:
        INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
        tmp.append(INorm.reshape((-1, self.problem.DI)))
    InewNorms=np.vstack([tmp[i] for i in range(ntsn)]).reshape((ntsn,self.problem.DI))

    # convert the parameter spaces to the normalized spaces
    PSoptNorms = self.problem.PS.transform(PSopt)
    columns = []
    for j in range(self.problem.DP):
        columns.append([])
    for i in range(ntso):
        for j in range(self.problem.DP):
            columns[j].append(PSoptNorms[i][j])
    PSoptNorms = []
    for j in range(self.problem.DP):
        PSoptNorms.append(np.asarray(columns[j]).reshape((ntso, -1)))



    # Predict optimums of new tasks
    for k in range(self.problem.DP):
        K = GPy.kern.RBF(input_dim=self.problem.DI)
        M = GPy.models.GPRegression(INorms, PSoptNorms[k], K)
        # M.optimize_restarts(num_restarts = 10, robust=True, verbose=False, parallel=False, num_processes=None, messages="False")
        M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust=True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] is not None and kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = 'lbfgs', start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
        MSopt.append(M)

    aprxoptsNorm=np.hstack([MSopt[k].predict_noiseless(InewNorms)[0] for k in range(self.problem.DP)])  # the index [0] is the mean value, [1] is the variance
    aprxoptsNorm=np.minimum(aprxoptsNorm,(1-1e-12)*np.ones((ntsn,self.problem.DP)))
    aprxoptsNorm=np.maximum(aprxoptsNorm,(1e-12)*np.ones((ntsn,self.problem.DP)))
    # print('aprxoptsNorm',aprxoptsNorm,type(aprxoptsNorm))
    aprxopts = self.problem.PS.inverse_transform(aprxoptsNorm)
    # print('aprxopts',aprxopts,type(aprxopts),type(aprxopts[0]))


    aprxoptsNormList=[]
    # TnewNormList=[]
    for i in range(ntsn):
        aprxoptsNormList.append([aprxoptsNorm[i,:]])  # this makes sure for each task, there is only one sample parameter set
        # InewNormList.append(InewNorms[i,:])

    t1 = time.time_ns()
    O = self.computer.evaluate_objective(problem = self.problem, I = InewNorms, P =aprxoptsNormList, options = kwargs)
    t2 = time.time_ns()
    time_fun = time_fun + (t2-t1)/1e9

    #        print(aprxopts)
    #        pickle.dump(aprxopts, open('TLA1.pkl', 'w'))

    t4 = time.time_ns()
    stats['time_total'] = (t4-t3)/1e9
    stats['time_fun'] = time_fun

    return (aprxopts,O,stats)

