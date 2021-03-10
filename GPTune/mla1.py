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


def MLA1(self, NS, NS1 = None, NI = None, Igiven = None, **kwargs):

    print('\n\n\n------Starting MLA with %d tasks and %d samples each '%(NI,NS))
    stats = {
        "time_total": 0,
        "time_sample_init": 0,
        "time_fun": 0,
        "time_search": 0,
        "time_model": 0
    }
    time_fun=0
    time_sample_init=0
    time_search=0
    time_model=0

    np.set_printoptions(suppress=False,precision=4)

#    if (self.data.P is not None and len(self.data.P[0])>=NS):
#        print('self.data.P[0])>=NS, no need to run MLA. Returning...')
#        return (copy.deepcopy(self.data), None,stats)

    t3 = time.time_ns()

    t1 = time.time_ns()

    options1 = copy.deepcopy(self.options)
    kwargs.update(options1)

    """ Multi-task Learning Autotuning """
    
    if(Igiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
        self.data.I = Igiven 

########## normalize the data as the user always work in the original space
    self.data = self.data.normalized()
    
#        if (self.mpi_rank == 0):

    sampler = eval(f'{kwargs["sample_class"]}()')
    if (self.data.I is None):

        if (NI is None):
            raise Exception("Number of problems to be generated (NI) is not defined")

        check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = True, kwargs = kwargs)
        self.data.I = sampler.sample_inputs(n_samples = NI, IS = self.problem.IS, check_constraints = check_constraints, **kwargs)
        # print("riji",type(self.data.I),type(self.data.I[0]))
        self.data.D = [{}] * NI
    else:
        if (self.data.D is None):
            self.data.D = [{}] * len(self.data.I)
    
    if (self.data.P is not None and len(self.data.P) !=len(self.data.I)):
        raise Exception("len(self.data.P) !=len(self.data.I)")
    
    if (self.data.P is None):
        if (NS1 is not None and NS1>NS):
            raise Exception("NS1>NS")
            
        if (NS1 is None):
            NS1 = min(NS - 1, 3 * self.problem.DP) # General heuristic rule in the litterature

        check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
        self.data.P = sampler.sample_parameters(n_samples = NS1, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
#        #XXX add the info of problem.models here
#        for P2 in P:
#            for x in P2:
#                x = np.concatenate(x, np.array([m(x) for m in self.problems.models]))
    # print("good?")
    
    if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
        raise Exception("len(self.data.O) !=len(self.data.I)")
    
    t2 = time.time_ns()
    time_sample_init = time_sample_init    + (t2-t1)/1e9

    t1 = time.time_ns()
    if (self.data.O is None):
        self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, options = kwargs) 
    t2 = time.time_ns()
    time_fun = time_fun + (t2-t1)/1e9
    # print(self.data.O)
    # print("good!")
#        if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
#            mpi_comm.bcast(self.data, root=0)
#
#    else:
#
#        self.data = mpi_comm.bcast(None, root=0)
    # mpi4py.MPI.COMM_WORLD.Barrier()
    print(self.data.O)

    modelers  = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
    searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')
    optiter = len(self.data.P[0])
    cpt = 0
    while (optiter < NS):# YL: each iteration adds 1 (if single objective) or at most kwargs["search_more_samples"] (if multi-objective) sample until total #sample reaches NS
    # for optiter in range(NS - len(self.data.P[0])): 

        if(self.problem.models_update is not None):
            ########## denormalize the data as the user always work in the original space
            tmpdata = self.data.originalized()
            self.problem.models_update(tmpdata)
            self.data.D = tmpdata.D

        # print("riji",type(self.data.I),type(self.data.I[0]))
        print("MLA iteration: ",optiter)
        optiter = optiter + 1
        t1 = time.time_ns()
        for o in range(self.problem.DO):
            tmpdata = copy.deepcopy(self.data)
            tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]
            if(self.problem.models is not None):
                for i in range(len(tmpdata.P)):
                    points0 = tmpdata.D[i]
                    t = tmpdata.I[i]
                    I_orig = self.problem.IS.inverse_transform(np.array(t, ndmin=2))[0]
                    points1 = {self.problem.IS[k].name: I_orig[k] for k in range(self.problem.DI)}
                    modeldata=[]
                    for p in range(len(tmpdata.P[i])):
                        x = tmpdata.P[i][p]
                        x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                        points = {self.problem.PS[k].name: x_orig[k] for k in range(self.problem.DP)}
                        points.update(points1)
                        points.update(points0)
                        modeldata.append(self.problem.models(points))
                    modeldata=np.array(modeldata)
                    tmpdata.P[i] = np.hstack((tmpdata.P[i],modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space 
            if (cpt == 0):
                modelers[o].train(data = tmpdata, **kwargs)
        
        t2 = time.time_ns()
        time_model = time_model + (t2-t1)/1e9

        t1 = time.time_ns()
        res = searcher.search_multitask(data = self.data, models = modelers, **kwargs)
        more_samples=NS-len(self.data.P[0]) # YL: this makes sure P has the same length across all tasks
        for x in res:
            more_samples=min(more_samples,x[1][0].shape[0])
        newdata = Data(problem = self.problem, I = self.data.I, D = self.data.D)
        newdata.P = [x[1][0][0:more_samples,:] for x in res]
        # print(more_samples,newdata.P)
        t2 = time.time_ns()
        time_search = time_search + (t2-t1)/1e9
#XXX add the info of problem.models here

#            if (self.mpi_rank == 0):

        t1 = time.time_ns()
        newdata.O = self.computer.evaluate_objective(problem = self.problem, I = newdata.I, P = newdata.P, D = newdata.D, options = kwargs)
        t2 = time.time_ns()
        time_fun = time_fun + (t2-t1)/1e9
#                if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
#                    mpi_comm.bcast(newdata.O, root=0)
#
#            else:
#
#                newdata.O = mpi_comm.bcast(None, root=0)
        self.data.merge(newdata)
    
########## denormalize the data as the user always work in the original space
    self.data = self.data.originalized()

    t4 = time.time_ns()
    stats['time_total'] = (t4-t3)/1e9
    stats['time_fun'] = time_fun
    stats['time_model'] = time_model
    stats['time_search'] = time_search
    stats['time_sample_init'] = time_sample_init
    
    
    return (copy.deepcopy(self.data), modelers,stats)

