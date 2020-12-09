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



def MLA(self, NS, NS1 = None, NI = None, Igiven = None, **kwargs):

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
    NSmin=0  
    if (self.data.P is not None):
        NSmin = min(map(len, self.data.P)) # the number of samples per task in existing tuning data can be different
                    
    if (self.data.P is not None and NSmin>=NS and self.data.O is not None):
        print('NSmin>=NS, no need to run MLA. Returning...')
        return (copy.deepcopy(self.data), None,stats)

    t3 = time.time_ns()

    t1 = time.time_ns()

    options1 = copy.deepcopy(self.options)
    kwargs.update(options1)

    """ Multi-task Learning Autotuning """

    if (NS1 is not None and NS1>NS):
        raise Exception("NS1>NS")
    if (NS1 is None):
        NS1 = min(NS - 1, 3 * self.problem.DP) # General heuristic rule in the litterature


    if(Igiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
        self.data.I = Igiven


########## normalize the data as the user always work in the original space
    if self.data.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
        tmp=[]
        for x in self.data.P:
            xNorm = self.problem.PS.transform(x)
            tmp.append(xNorm)
        self.data.P=tmp
    if self.data.I is not None: # from a list of lists to a 2D numpy array
        self.data.I = self.problem.IS.transform(self.data.I)

    if (self.data.O is None and self.data.P is not None and self.data.I is not None): # tuning parameters and task parameters are given, but the output is none
        self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, options = kwargs)
            

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
            self.data.D = [{}] * NI

    if (self.data.P is not None and len(self.data.P) !=len(self.data.I)):
        raise Exception("len(self.data.P) !=len(self.data.I)")

    if (NSmin<NS1):
        check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
        tmpP = sampler.sample_parameters(n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
        if (NSmin>0):
            for i in range(len(self.data.P)):
                NSi = self.data.P[i].shape[0]
                tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data


#            #XXX add the info of problem.models here
#            for P2 in P:
#                for x in P2:
#                    x = np.concatenate(x, np.array([m(x) for m in self.problems.models]))
    # print("good?")

    if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
        raise Exception("len(self.data.O) !=len(self.data.I)")

    t2 = time.time_ns()
    time_sample_init = time_sample_init + (t2-t1)/1e9

    t1 = time.time_ns()
    if (NSmin<NS1):
        tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, options = kwargs)
        if(NSmin==0): # no existing tuning data is available
            self.data.O = tmpO
            self.data.P = tmpP	
        else:
            for i in range(len(self.data.P)):
                self.data.P[i] = np.vstack((self.data.P[i],tmpP[i]))
                self.data.O[i] = np.vstack((self.data.O[i],tmpO[i]))

    t2 = time.time_ns()
    time_fun = time_fun + (t2-t1)/1e9
    # print(self.data.O)
    # print("good!")
#            if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
#                mpi_comm.bcast(self.data, root=0)
#
#        else:
#
#            self.data = mpi_comm.bcast(None, root=0)
    # mpi4py.MPI.COMM_WORLD.Barrier()
    modelers  = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
    searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')
    optiter = 0
    NSmin = min(map(len, self.data.P)) 
    while NSmin<NS:# YL: each iteration adds 1 (if single objective) or at most kwargs["search_more_samples"] (if multi-objective) sample until total #sample reaches NS

        if(self.problem.models_update is not None):
            ########## denormalize the data as the user always work in the original space
            tmpdata = copy.deepcopy(self.data)
            if tmpdata.I is not None:    # from 2D numpy array to a list of lists
                tmpdata.I = self.problem.IS.inverse_transform(tmpdata.I)
            if tmpdata.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
                tmp=[]
                for x in tmpdata.P:
                    xOrig = self.problem.PS.inverse_transform(x)
                    tmp.append(xOrig)
                tmpdata.P=tmp
            self.problem.models_update(tmpdata)
            self.data.D = tmpdata.D

        newdata = Data(problem = self.problem, I = self.data.I, D = self.data.D)
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
            for i in range(len(tmpdata.P)):   # LCM requires the same number of samples per task, so use the first NSmin samples
                tmpdata.O[i] = tmpdata.O[i][0:NSmin,:]
                tmpdata.P[i] = tmpdata.P[i][0:NSmin,:]
            # print(tmpdata.P[0])
            modelers[o].train(data = tmpdata, **kwargs)
            if self.options['verbose'] == True and self.options['model_class'] == 'Model_LCM' and len(self.data.I)>1:
                C = modelers[o].M.kern.get_correlation_metric()
                print("The correlation matrix C is \n", C)
            elif self.options['verbose'] == True and self.options['model_class'] == 'Model_GPy_LCM' and len(self.data.I)>1:
                C = modelers[o].get_correlation_metric(len(self.data.I))
                print("The correlation matrix C is \n", C)
            
            
            
            

        t2 = time.time_ns()
        time_model = time_model + (t2-t1)/1e9

        t1 = time.time_ns()
        res = searcher.search_multitask(data = self.data, models = modelers, **kwargs)

        newdata.P = [x[1][0] for x in res]
        for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
            NSi = self.data.P[i].shape[0]
            newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]            
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
        NSmin = min(map(len, self.data.P))

########## denormalize the data as the user always work in the original space
    if self.data.I is not None:    # from 2D numpy array to a list of lists
        self.data.I = self.problem.IS.inverse_transform(self.data.I)
    if self.data.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
        tmp=[]
        for x in self.data.P:
            xOrig = self.problem.PS.inverse_transform(x)
            tmp.append(xOrig)
        self.data.P=tmp

    t4 = time.time_ns()
    stats['time_total'] = (t4-t3)/1e9
    stats['time_fun'] = time_fun
    stats['time_model'] = time_model
    stats['time_search'] = time_search
    stats['time_sample_init'] = time_sample_init


    return (copy.deepcopy(self.data), modelers,stats)


