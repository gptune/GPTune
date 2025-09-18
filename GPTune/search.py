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
import concurrent
from concurrent import futures
import sys
import abc
from typing import Collection
import numpy as np
import scipy as sp
import functools
from joblib import *

import copy
from problem import Problem
from computer import Computer
from options import Options
from data import Data
from model import Model
from sample import *

from pathlib import Path
import importlib
from sys import platform as _platform


class Search(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer, options: Options, models_transfer=None):
        self.problem = problem
        self.computer = computer
        self.options = options
        self.models_transfer = models_transfer

    @abc.abstractmethod
    def search(self, data : Data, models : Collection[Model], tid : int, **kwargs) -> np.ndarray:

        raise Exception("Abstract method")

    def search_multitask(self, data : Data, models : Collection[Model], tids : Collection[int] = None, i_am_manager : bool = True, **kwargs) -> Collection[np.ndarray]:

        if (tids is None):
            tids = list(range(data.NI))
        flag=0
        for i in range(self.problem.DO):
            if models is not None and models[i].mf is not None:
                flag=1
        if flag==1:
            print('Warning: there is currently no good way of spawning the mean_function, so distributed_memory_parallelism is disabled for the search!')

        if (kwargs['distributed_memory_parallelism'] and i_am_manager and flag==0):   # the pgymo install on mac os seems buggy if search is not spawned
            import mpi4py
            nproc = min(kwargs['search_multitask_processes'],data.NI)
            npernode = int(self.computer.cores/kwargs['search_multitask_threads'])
            mpi_comm = self.computer.spawn(__file__, nproc=nproc, nthreads=kwargs['search_multitask_threads'], npernode=npernode, kwargs=kwargs) # XXX add args and kwargs
            kwargs_tmp = kwargs
            if "mpi_comm" in kwargs_tmp:
                del kwargs_tmp["mpi_comm"]   # mpi_comm is not picklable
            _ = mpi_comm.bcast((self, data, models, tids, kwargs_tmp), root=mpi4py.MPI.ROOT)
            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()
            res=[]
            for p in range(int(nproc)):
                res = res + tmpdata[p]


        elif (kwargs['shared_memory_parallelism']):
            #with concurrent.futures.ProcessPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
                # fun = functools.partial(self.search, data = data, models = models, kwargs = kwargs)
                # res = list(executor.map(fun, tids, timeout=None, chunksize=1))

                def fun(tid):
                    return self.search(data=data,models = models, tid =tid, kwargs = kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize=1))
        else:
            fun = functools.partial(self.search, data, models, kwargs = kwargs)
            res = list(map(fun, tids))

        # print(res)

        # check if there are duplicated samples
        if data.P is not None:
            for res_ in res:
                tid = res_[0]
                x = res_[1][0]
                tmp = x
                duplicate = False
                for x_ in data.P[tid]:
                    x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                    x_orig_ = self.problem.PS.inverse_transform(np.array(x_, ndmin=2))[0]
                    if x_orig == x_orig_:
                        duplicate = True
                        print ("duplicated sample: ", x)
                        break

                while duplicate == True:
                    duplicate = False
                    # generate random sample if the sample already has duplicates
                    x = np.random.rand(len(tmp[0]))
                    print ("generate random sample: ", x)
                    print ("generate random sample (orig): ", self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0])
                    res_[1][0] = np.array([x.tolist()], ndmin=2)
                    for x_ in data.P[tid]:
                        x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                        x_orig_ = self.problem.PS.inverse_transform(np.array(x_, ndmin=2))[0]
                        if x_orig == x_orig_:
                            duplicate = True
                            break
        res.sort(key = lambda x : x[0])
        return res

class SurrogateProblem(object):

    def __init__(self, problem, computer, data, models, options, tid, models_transfer):   # data is in the normalized space, IOrig and POrig are then generated in the original space

        self.problem = problem
        self.computer = computer
        self.data = data
        self.models = models
        self.models_last = models
        self.options = options

        self.tid = tid

        self.D     = self.data.D[tid]
        self.IOrig = self.problem.IS.inverse_transform(np.array(self.data.I[tid], ndmin=2))[0]
        # if (self.options['verbose']):
        #     print ("self.IOrig: ", self.IOrig)

        if self.data.P is not None and len(self.data.P[tid]) > 0:
            self.POrig = self.problem.PS.inverse_transform(np.array(self.data.P[tid], ndmin=2))
        else:
            self.POrig = [] # self.POrig = [[]]
        # if (self.options['verbose']):
        #     print ("self.POrig: ", self.POrig)

        self.models_transfer = models_transfer
        if (self.models != None and self.models_transfer != None and self.options['TLA_method'] == 'Regression'):
            self.models_weights = self.compute_weights()
            if self.options['regression_logging'] == True:
                with open(self.options['regression_log_name'], "a") as f_out:
                    for i in range(len(self.models_weights)):
                        if i > 0:
                            f_out.write(",")
                        f_out.write(str(self.models_weights[i]))
                    f_out.write("\n")

        #### precompute the adjusted bounds and Pereto Front given all the existing samples
        if self.options['search_af'] == 'UCB-HVI': # YC: the following computation is needed only for UCB-HVI.
        # YC: Note: Using UCB-HVI in TLA_I can cause an error, because TLA_I can fall into this with 0 samples for the target task, so it can't compute the upper bound below.
            dataO = copy.deepcopy(self.data.O)
            A = []
            B = []
            for o in range(self.problem.DO):
                lower_bound, upper_bound = self.problem.OS.bounds[o]
                if(math.isinf(upper_bound)):
                    upper_bound=dataO[self.tid][:,o].max()
                else:
                    dataO[self.tid][:,o] = np.where(dataO[self.tid][:,o] < upper_bound, dataO[self.tid][:,o], upper_bound)

                if(self.problem.OS[o].optimize== False): # if not optimized, set the data on that dimension to be constant
                    dataO[self.tid][:,o]=upper_bound

                A.append(lower_bound)
                B.append(upper_bound)

            self.A=np.array(A).reshape(self.problem.DO,)
            self.B=np.array(B).reshape(self.problem.DO,)
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            PF_idx = NonDominatedSorting(method="fast_non_dominated_sort").do(dataO[self.tid], only_non_dominated_front=True)
            PF = [dataO[self.tid][i,:] for i in PF_idx]
            self.PF=np.array(PF).reshape(len(PF_idx),self.problem.DO)

    def compute_weights(self):
        #This function computes the weights for surrogate models to be combined.
        #The formula that defines the regression, which determines the weights:
        #In the setting where we want to see which surrogate contributes the most to the maximum.
        #For the j-th model out of N surrogates, suppose that (xmax,ymax) is the observed maximum.
        #y_j-ymax=\sum_{i=1}^{d} w_i mean_i(x_j)-mean_i(xmax)+\epsilon_{j},j=1,\cdots,N
        #LHS: difference to the response maximum y-ymax as response variable.
        #RHS: a linear model using x-xmax as predictors.
        for o in range(self.problem.DO):
            if len(self.data.O[self.tid][:,o]) == 1:
                models_weights = [1]
                for model_transfer in self.models_transfer:
                    models_weights.append(1)
                models_weights = np.array(models_weights)
                models_weights = models_weights/np.sum(models_weights)
                print ("models_weights: ", models_weights)
                return models_weights

            ymin = self.data.O[self.tid][:,o].min()
            ymin_index = self.data.O[self.tid][:,o].tolist().index(ymin)
            x_list = self.data.P[self.tid][:]
            x_star = x_list[ymin_index]
            y_list = self.data.O[self.tid][:,o]
            if self.options['TLA_method'] == 'Regression_No_Scale':
                LHS = [(-1.0*float(y_elem))-(-1.0*(ymin)) for y_elem in y_list]
            else:
                LHS = [(-1.0*float(y_elem/ymin))-(-1.0*(ymin/ymin)) for y_elem in y_list]
            print ("LHS: ", LHS)
            RHS = []
            for x_sample in x_list:
                point = self.D
                point.update({self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)})
                point_x_sample = point.copy()
                point_x_star = point.copy()
                for k in range(self.problem.DP):
                    point_x_sample[self.problem.PS[k].name] = x_sample[k]
                for k in range(self.problem.DP):
                    point_x_star[self.problem.PS[k].name] = x_star[k]

                x_sample_orig = self.problem.PS.inverse_transform(np.array(x_sample, ndmin=2))[0]
                x_star_orig = self.problem.PS.inverse_transform(np.array(x_star, ndmin=2))[0]
                point_x_sample_orig = point.copy()
                point_x_star_orig = point.copy()
                for k in range(self.problem.DP):
                    point_x_sample_orig[self.problem.PS[k].name] = x_sample_orig[k]
                for k in range(self.problem.DP):
                    point_x_star_orig[self.problem.PS[k].name] = x_star_orig[k]

                print ("point_x_sample: ", point_x_sample)
                print ("point_x_star: ", point_x_star)
                print ("point_x_sample_orig: ", point_x_sample_orig)
                print ("point_x_star_orig: ", point_x_star_orig)

                RHS_row = []
                (mu, var) = self.models[o].predict_last(x_sample, tid=self.tid)
                mu = mu[0][0]
                (mu_star, var_star) = self.models[o].predict_last(x_star, tid=self.tid)
                mu_star = mu_star[0][0]
                if self.options['TLA_method'] == 'Regression_No_Scale':
                    RHS_elem = (-1.0*float(mu))-(-1.0*(mu_star))
                else:
                    RHS_elem = (-1.0*float(mu/mu_star))-(-1.0*(mu_star/mu_star))
                print ("RHS_elem (current task): ", RHS_elem)
                RHS_row.append(RHS_elem)
                for model_transfer in self.models_transfer:
                    ret = model_transfer(point_x_sample_orig)
                    print ("RET: ", ret)
                    mu = ret[self.problem.OS[o].name][0][0]
                    ret = model_transfer(point_x_star_orig)
                    mu_star = max(1e-18, ret[self.problem.OS[o].name][0][0])
                    if self.options['TLA_method'] == 'Regression_No_Scale':
                        RHS_elem = (-1.0*float(mu))-(-1.0*(mu_star))
                    else:
                        RHS_elem = (-1.0*float(mu/mu_star))-(-1.0*(mu_star/mu_star))
                    #print ("RHS_elem: ", RHS_elem)
                    RHS_row.append(RHS_elem)
                RHS.append(RHS_row)
            print ("RHS: ", RHS)

            LHS = np.array(LHS)
            RHS = np.array(RHS)
            #solve the linear system defined by above.
            try:
                LSTSQ_SOL = np.linalg.lstsq(RHS, LHS)
                models_weights = LSTSQ_SOL[0]
            except:
                print ("unexpected error from np.linalg.lstsq routine, manually assign the same weight for this sampling point")
                models_weights = [1.0 for i in range(len(self.models_transfer)+1)]

            print ("models_weights: ", models_weights)
            print ("models_weights_sum: ", np.sum(models_weights))
            models_weights_normalized = models_weights / np.sum(models_weights)
            print ("models_weights_normalized: ", models_weights_normalized)
            return models_weights_normalized

    def get_nobj(self):
        if(self.options['search_algo']=='pso' or self.options['search_algo']=='cmaes'):
            return 1
        else:
            return self.problem.DO

    def get_bounds(self):
        if(self.options['search_af']=='q-UCB' or self.options['search_af']=='q-EI'): # To the fitness function, input dimension is DP*search_more_samples instead of DP to use the evolutionary algorithm
            DP = self.problem.DP*self.options['search_more_samples']
        else:
            DP = self.problem.DP
        return ([0. for i in range(DP)], [1. for  i in range(DP)])

    def is_dominated(self, x, S):
        is_dom = False
        for pt in S:
            if np.all(pt <= x):
                is_dom = True
                break
        return is_dom

    # Acquisition function
    def af(self, x):

        if self.options['search_af'] == 'UCB-HVI':
            uhvi_pt = np.empty(self.problem.DO)
            for o in range(self.problem.DO):
                # print(o,self.models[o].M.kern.lengthscale)
                (mu, var) = self.models[o].predict(x, tid=self.tid)
                var = max(1e-18, var[0][0])
                uhvi_pt[o] = (mu - np.sqrt(self.options['search_ucb_beta']* var))
            uhvi_pt = np.where(uhvi_pt > self.A, uhvi_pt, self.A)
            for o in range(self.problem.DO):
                if(self.problem.OS[o].optimize== False): # if not optimized, set the data on that dimension to be constant
                    uhvi_pt[o]=self.B[o]
            if self.is_dominated(uhvi_pt, self.PF) or np.any(uhvi_pt > self.B):
                uhvi = 0
            else:
                #add the uhvi_pt to the list of points
                points = np.vstack((self.PF,np.atleast_2d(uhvi_pt)))
                if importlib.util.find_spec("pygmo") is not None:
                    import pygmo as pg
                    hv = pg.hypervolume(points)
                    #calculate the exclusive contribution to the hypervolume from our point
                    uhvi = hv.exclusive(len(points)-1, self.B)
                else:
                    from pymoo.indicators.hv import HV
                    ref_point = np.array(self.B)
                    ind = HV(ref_point=ref_point)
                    uhvi = ind(points) - ind(self.PF)
                    # print(uhvi,ind(points),ind(self.PF),uhvi_pt,self.B,self.A)

            
            return [-uhvi]
        else:
            AF=[]
            for o in range(self.problem.DO):
                optimize = self.problem.OS[o].optimize
                # YC: If PSO is given by user, for no-optimize objectives we can simply ignore the objectives' outputs in AF product
                if (self.options['search_algo']=='pso' and optimize == False):
                    #print ("o: ", self.problem.OS[o].name, "is not optimize")
                    continue
                # YC: If NSGA2 is given by user (e.g. more than three objectives tuning and one no-optimize objective), for no-optimize objectives we still need to return some value for running NSGA2. Maybe we can fix this later.
                elif (optimize == False):
                    #print ("o: ", self.problem.OS[o].name, "is not optimize")
                    AF.append(0)
                else:
                    if self.models_transfer == None:
                        if self.data.O == None:
                            (mu, var) = self.models[o].predict(x, tid=self.tid)
                            mu = mu[0][0]
                            var = max(1e-18, var[0][0])
                            AF.append(1.0/mu)
                        else:
                            if self.options['search_af'] == 'EI':
                                ymin = self.data.O[self.tid][:,o].min()
                                (mu, var) = self.models[o].predict(x, tid=self.tid)
                                mu = mu[0][0]
                                var = max(1e-18, var[0][0])
                                std = np.sqrt(var)                            
                                chi = (ymin - mu -self.options['search_ei_alpha']) / std
                                Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
                                phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
                                AF.append(-((ymin - mu -self.options['search_ei_alpha']) * Phi + std * phi))
                            elif self.options['search_af'] == 'UCB': # as we are minimizing af, use mu - sqrt(beta)std (LCB) instead of mu + sqrt(beta)std (UCB)
                                (mu, var) = self.models[o].predict(x, tid=self.tid)
                                mu = mu[0][0]
                                var = max(1e-18, var[0][0])
                                std = np.sqrt(var)                            
                                AF.append(mu - np.sqrt(self.options['search_ucb_beta'])*std)
                            elif self.options['search_af'] == 'MSPE': #min square prediction error as used in cGP                          
                                X_joint = np.vstack((x,np.array(self.data.P[self.tid], ndmin=2)))
                                (mu_cross, sigma_cross) = self.models[o].predict(X_joint, tid=self.tid, full_cov=True)
                                sigma = sigma_cross[0:1,0:1]
                                sigma_obs = sigma_cross[1:sigma_cross.shape[1],1:sigma_cross.shape[1]]
                                sigma_cross = sigma_cross[0:1,1:sigma_cross.shape[1]]
                                sigma_cross = sigma_cross.reshape(-1,1).T                            
                                mspe = (sigma - sigma_cross @ sigma_obs @ sigma_cross.T)/(X_joint.shape[0]-1)
                                AF.append(mspe[0][0])
                            elif self.options['search_af'] == 'q-UCB': #multi-point UCB function in the paper "The reparameterization trick for acquisition functions", 2017
                                (mu_cross, sigma_cross) = self.models[o].predict(x, tid=self.tid, full_cov=True)
                                # print(mu_cross,sigma_cross,x.shape[0])
                                
                                if(x.shape[0]==1):
                                    tmp = mu_cross-np.sqrt(sigma_cross*self.options['search_ucb_beta'] * np.pi/2)
                                    tmp = tmp[0][0]
                                else:
                                    n=1000*self.options['search_more_samples'] # number of Monte Carlo samples, this is from heuristics
                                    tmp=0
                                    sigma_cross0 = sigma_cross *self.options['search_ucb_beta'] * np.pi/2                                    
                                    
                                    jitter=self.options['model_jitter']
                                    flag=0
                                    for i in range(self.options['model_max_jitter_try']):
                                        try:
                                            sigma_cross = np.linalg.cholesky(sigma_cross0+np.diag(np.ones((1,self.options['search_more_samples']))*jitter))
                                            flag=1
                                            break
                                        except:
                                            jitter=jitter*10
                                    if(flag==0):
                                        raise Exception("sigma_cross not SPD after jittering")

                                    if(True):
                                        zk0 = np.random.randn(x.shape[0],n)
                                        mat=- np.absolute(sigma_cross @ zk0)
                                        mat+=mu_cross
                                        tmp = np.amax(mat,axis = 0)
                                        tmp =np.sum(tmp)
                                    else:
                                        for i in range(n):
                                            zk = np.random.randn(x.shape[0])
                                            uaL = np.amax(mu_cross - np.absolute(sigma_cross @ zk))/n
                                            tmp = tmp + uaL
                                            # print(i,uaL,np.absolute(sigma_cross @ zk))
                                # print(tmp)
                                # sys.exit()
                                AF.append(tmp)
                            elif self.options['search_af'] == 'q-EI': #multi-point EI function in the paper "The reparameterization trick for acquisition functions", 2017
                                ymin = self.data.O[self.tid][:,o].min()
                                (mu_cross, sigma_cross0) = self.models[o].predict(x, tid=self.tid, full_cov=True)
                                
                                n=1000*self.options['search_more_samples'] # number of Monte Carlo samples, this is heuristics
                                tmp=0

                                jitter=self.options['model_jitter']
                                flag=0
                                for i in range(self.options['model_max_jitter_try']):
                                    try:
                                        sigma_cross = np.linalg.cholesky(sigma_cross0+np.diag(np.ones((1,self.options['search_more_samples']))*jitter))
                                        flag=1
                                        break
                                    except:
                                        jitter=jitter*10
                                if(flag==0):
                                    raise Exception("sigma_cross not SPD after jittering")
                                
                                if(True):
                                    zk0 = np.random.randn(x.shape[0],n)
                                    mat = sigma_cross @ zk0
                                    tmpcol = ymin - mu_cross -self.options['search_ei_alpha']
                                    mat += tmpcol
                                    tmp = np.amax(mat,axis = 0)
                                    tmp =np.sum(x for x in tmp if x>0)
                                else:
                                    for i in range(n):
                                        zk = np.random.randn(x.shape[0])
                                        uaL = max(0.0,np.amax(ymin - mu_cross -self.options['search_ei_alpha'] + sigma_cross @ zk )/n)
                                        tmp = tmp + uaL
                                        # print(i,uaL,np.absolute(sigma_cross @ zk))

                                AF.append(-tmp)
                            else:
                                raise Exception("unknown aquicision function %s"%(self.options['search_af']))
                            # AF.append(mu)
                    elif self.models_transfer is not None and self.models is None:
                        xi0 = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
                        xi=xi0[0]

                        if (any(xx==xi for xx in self.POrig)):
                            cond = False
                        else:
                            point0 = self.D
                            point2 = {self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)}
                            point  = {self.problem.PS[k].name: xi[k] for k in range(self.problem.DP)}
                            point.update(point0)
                            point.update(point2)
                            cond = self.computer.evaluate_constraints(self.problem, point)

                        mu_transfer = 0
                        var_transfer = 1
                        num_models_transfer = len(self.models_transfer)
                        for i in range(num_models_transfer):
                            model_transfer = self.models_transfer[i]
                            ret = model_transfer(point)
                            mu_transfer += 1.0/len(self.models_transfer)*ret[self.problem.OS[o].name][0][0]
                            try:
                                var_transfer_ = math.pow(max(1e-18, ret[self.problem.OS[o].name+"_var"][0][0]), float(1.0/num_models_transfer))
                            except:
                                var_transfer_ = 1
                            var_transfer *= var_transfer_
                        var = max(1e-18, var_transfer)
                        AF.append(1.0/mu_transfer)
                    elif self.options['TLA_method'] == 'Regression':
                        xi0 = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
                        xi=xi0[0]

                        if (any(xx==xi for xx in self.POrig)):
                            cond = False
                        else:
                            point0 = self.D
                            point2 = {self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)}
                            point  = {self.problem.PS[k].name: xi[k] for k in range(self.problem.DP)}
                            point.update(point0)
                            point.update(point2)
                            cond = self.computer.evaluate_constraints(self.problem, point)

                        ymin = self.data.O[self.tid][:,o].min()
                        (mu, var) = self.models[o].predict(x, tid=self.tid)
                        mu_transfer = 0
                        var_transfer = 1

                        for i in range(len(self.models_transfer)):
                            model_transfer = self.models_transfer[i]
                            ret = model_transfer(point)
                            mu_transfer += self.models_weights[i+1]*ret[self.problem.OS[o].name][0][0]
                            try:
                                var_transfer_ = math.pow(max(1e-18, ret[self.problem.OS[o].name+"_var"][0][0]), self.models_weights[i+1])
                            except:
                                var_transfer_ = 1
                            var_transfer *= var_transfer_
                        mu = self.models_weights[0]*mu[0][0] + mu_transfer
                        var_transfer *= math.pow(max(1e-18, var[0][0]), self.models_weights[0])
                        var = max(1e-18, var_transfer)
                        std = np.sqrt(var)
                        chi = (ymin - mu-self.options['search_ei_alpha']) / std
                        Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
                        phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
                        AF.append(-((ymin - mu-self.options['search_ei_alpha']) * Phi + std * phi))
                        # AF.append(mu)
                    elif self.options['TLA_method'] == 'LCM' or self.options['TLA_method'] == 'LCM_BF':
                        ymin = self.data.O[self.tid][:,o].min()
                        (mu, var) = self.models[o].predict(x, tid=self.tid)
                        mu = mu[0][0]
                        var = max(1e-18, var[0][0])
                        std = np.sqrt(var)
                        chi = (ymin - mu-self.options['search_ei_alpha']) / std
                        Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
                        phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
                        AF.append(-((ymin - mu-self.options['search_ei_alpha']) * Phi + std * phi))
                        # AF.append(mu)
                    elif self.options['TLA_method'] == 'Sum':
                        xi0 = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
                        xi=xi0[0]

                        if (any(xx==xi for xx in self.POrig)):
                            cond = False
                        else:
                            point0 = self.D
                            point2 = {self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)}
                            point  = {self.problem.PS[k].name: xi[k] for k in range(self.problem.DP)}
                            point.update(point0)
                            point.update(point2)
                            cond = self.computer.evaluate_constraints(self.problem, point)

                        ymin = self.data.O[self.tid][:,o].min()
                        (mu, var) = self.models[o].predict(x, tid=self.tid)
                        mu_transfer = 0
                        var_transfer = 1
                        num_models_transfer = len(self.models_transfer)
                        for model_transfer in self.models_transfer:
                            ret = model_transfer(point)
                            mu_transfer += 1*ret[self.problem.OS[o].name][0][0]
                            try:
                                var_transfer_ = math.pow(max(1e-18, ret[self.problem.OS[o].name+"_var"][0][0]), float(1.0/(num_models_transfer+1)))
                            except:
                                var_transfer_ = 1
                            var_transfer *= var_transfer_
                        mu = mu[0][0] + mu_transfer
                        var_transfer *= math.pow(max(1e-18, var[0][0]), float(1.0/(num_models_transfer+1)))
                        var = max(1e-18, var_transfer)
                        std = np.sqrt(var)
                        chi = (ymin - mu-self.options['search_ei_alpha']) / std
                        Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
                        phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
                        AF.append(-((ymin - mu-self.options['search_ei_alpha']) * Phi + std * phi))
                        # AF.append(mu)
            return AF

            # #### YL: The following seems buggy for AF functions other than EI. I'm commenting this out as the idea of product of multiple AFs seems to not perform well.  
            # if(self.options['search_algo']=='pso' or self.options['search_algo']=='cmaes'):
            #     AF_prod = np.prod(AF)
            #     return [-1.0*AF_prod if AF_prod>0 else AF_prod]
            # else:
            #     return AF

    def fitness(self, x):   # x is the normalized space
        if(self.options['search_af']=='q-UCB' or self.options['search_af']=='q-EI'):
            x=np.array(x, ndmin=2).reshape(self.options['search_more_samples'],-1)

        xi0 = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
        xNorm = self.problem.PS.transform(xi0)
        CND = True
        modeldata=[]
        for xi in xi0:
            if (any(xx==xi for xx in self.POrig)):
                cond = False
                CND = False
            else:
                point0 = self.D
                point2 = {self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)}
                point  = {self.problem.PS[k].name: xi[k] for k in range(self.problem.DP)}
                point.update(point0)
                point.update(point2)
                # print("point", point)
                cond = self.computer.evaluate_constraints(self.problem, point)
                if(cond == False):
                    CND = False

            if (cond):
                if(self.problem.models is not None):    
                    if(self.options['distributed_memory_parallelism']== True):                
                        if(self.problem.driverabspath is not None):
                            modulename = Path(self.problem.driverabspath).stem  # get the driver name excluding all directories and extensions
                            sys.path.append(self.problem.driverabspath) # add path to sys
                            module = importlib.import_module(modulename) # import driver name as a module
                        else:
                            raise Exception('performance models require passing driverabspath to GPTune')
                        modeldata.append(module.models(point))
                    else:
                        modeldata.append(self.problem.models(point))                  
        if (CND):
            if(self.problem.models is not None):
                xNorm = np.hstack((xNorm,np.array(modeldata).reshape(self.options['search_more_samples'],1)))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
            # print(xNorm)  
            # print('I am here')
            # print(self.af(xNorm))
            # sys.exit()          
            # print("cond",cond,- self.af(x),'x',x,'xi',xi)
            #print ("AF: ", self.af(xNorm))
            return self.af(xNorm)
        else:
            # print("cond",cond,float("Inf"),'x',x,'xi',xi)
            if(self.problem.DO==1): # single objective optimizer
                return [self.options['search_bigval']]
            elif(self.options['search_algo']=='pso' or self.options['search_algo']=='cmaes'): 
                return [self.options['search_bigval']]
            else:
                return [self.options['search_bigval']]* self.problem.DO
    
    def obj_scipy(self, x):
        return self.fitness(x)[0]


from pymoo.core.problem import ElementwiseProblem
class MyProblemPyMoo(ElementwiseProblem):

    def __init__(self,n_var,n_obj,prob):
        super().__init__(n_var=n_var,n_obj=n_obj,n_constr=0,xl=np.array([0]*n_var),xu=np.array([1]*n_var))
        self.prob=prob

    def _evaluate(self, x, out, *args, **kwargs):
        fs = self.prob.fitness(x)
        out["F"] = fs


class SearchPyMoo(Search):

    def search(self, data : Data, models : Collection[Model], tid : int, **kwargs) -> np.ndarray:

        kwargs = kwargs['kwargs']

        print("searcher: ", kwargs["search_class"], "algorithm: ", kwargs["search_algo"])

        prob = SurrogateProblem(self.problem, self.computer, data, models, self.options, tid, self.models_transfer)

        if (kwargs['verbose']):
            print ("prob: ", prob)
        bestX = []


        if(self.problem.DO==1 or kwargs['search_af']=='UCB-HVI'): # single objective optimizer
            prob_pymoo = MyProblemPyMoo(self.problem.DP,1,prob)
            if('ga'==kwargs['search_algo']):
                from pymoo.algorithms.soo.nonconvex.ga import GA
                from pymoo.optimize import minimize
                algo = GA(pop_size = kwargs["search_pop_size"])
            elif('pso'==kwargs['search_algo']):   
                from pymoo.algorithms.soo.nonconvex.pso import PSO
                from pymoo.optimize import minimize
                algo = PSO(pop_size = kwargs["search_pop_size"])                
            else:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')

            bestX = []
            if kwargs['search_random_seed'] == None:
                res = minimize(prob_pymoo,algo,verbose=kwargs['verbose'],seed=1)
            else:
                seed = kwargs['search_random_seed']
                if data.P is not None:
                    for P_ in data.P:
                        seed += len(P_)
                res = minimize(prob_pymoo,algo,verbose=kwargs['verbose'],seed=seed)
            bestX.append(np.array(res.X).reshape(1, self.problem.DP))

        else:                   # multi objective
            prob_pymoo = MyProblemPyMoo(self.problem.DP,self.problem.DO,prob)
            if('nsga2'==kwargs['search_algo']):
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.optimize import minimize
                algo = NSGA2(pop_size = kwargs["search_pop_size"])
            elif('moead'==kwargs['search_algo']): 
                from pymoo.algorithms.moo.moead import MOEAD
                from pymoo.optimize import minimize
                from pymoo.factory import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", self.problem.DO, n_partitions=12)
                algo = MOEAD(ref_dirs, n_neighbors=15,prob_neighbor_mating=0.7)
            else:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')
            bestX = []
            res = minimize(prob_pymoo,algo,("n_gen", kwargs["search_gen"]),verbose=kwargs['verbose'],seed=1)
            firstn = min(int(kwargs['search_more_samples']),np.shape(res.X)[0])
            xss = res.X[0:firstn]
            bestX.append(xss)

        if (kwargs['verbose']):
            print(tid); sys.stdout.flush()
            print("bestX",bestX)
        return (tid, bestX)


class SearchPyGMO(Search):

    """
    XXX: This class, together with the underlying PyGMO only works on Intel and AMD CPUs.
    The reason is that PyGMO requires the Intel 'Thread Building Block' library to compile and execute.
    """

    def search(self, data : Data, models : Collection[Model], tid : int, **kwargs) -> np.ndarray:
        import pygmo as pg

        kwargs = kwargs['kwargs']

        print("searcher: ", kwargs["search_class"], "algorithm: ", kwargs["search_algo"])
        prob = SurrogateProblem(self.problem, self.computer, data, models, self.options, tid, self.models_transfer)

        if (kwargs['verbose']):
            print ("prob: ", prob)

        try:
            udi = eval(f'pg.{kwargs["search_udi"]}()')
        except:
            raise Exception('Unknown user-defined-island "{kwargs["search_udi"]}"')

        # if(self.problem.DO==1 or kwargs['search_algo']=='pso' or kwargs['search_algo']=='cmaes' or kwargs['search_algo']==['search_af'] == 'UCB-HVI'): # single objective optimizer
        if(self.problem.DO==1 ): # single objective optimizer
            try:
                algo = eval(f'pg.{kwargs["search_algo"]}(gen = kwargs["search_gen"])')
            except:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')
            bestX = []
            cond = False
            cpt = 0
            while (not cond and cpt < kwargs['search_max_iters']):
                if kwargs['search_random_seed'] == None:
                    archi = pg.archipelago(n = kwargs['search_threads'], prob = prob, algo = algo, udi = udi, pop_size = kwargs['search_pop_size'])
                else:
                    seed = kwargs['search_random_seed']
                    if data.P is not None:
                        for P_ in data.P:
                            seed += len(P_)
                    archi = pg.archipelago(n = kwargs['search_threads'], prob = prob, algo = algo, udi = udi, pop_size = kwargs['search_pop_size'], seed = seed)
                archi.evolve(n = kwargs['search_evolve'])
                archi.wait()
                champions_f = archi.get_champions_f()
                champions_x = archi.get_champions_x()
                indexes = list(range(len(champions_f)))
                indexes.sort(key=champions_f.__getitem__)
                for idx in indexes:
                    if (champions_f[idx] < self.options['search_bigval']):
                        cond = True
                        # bestX.append(np.array(self.problem.PS.inverse_transform(np.array(champions_x[idx], ndmin=2))[0]).reshape(1, self.problem.DP))
                        if(self.options['search_af']=='q-UCB' or self.options['search_af']=='q-EI'):
                            bestX.append(np.array(champions_x[idx]).reshape(self.options['search_more_samples'], self.problem.DP))
                        else:
                            bestX.append(np.array(champions_x[idx]).reshape(1, self.problem.DP))
                        break
                cpt += 1
        else:                   # multi objective
            try:
                algo = eval(f'pg.algorithm(pg.{kwargs["search_algo"]}(gen = kwargs["search_gen"]))')
            except:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')
            bestX = []
            cond = False
            cpt = 0
            while (not cond and cpt < kwargs['search_max_iters']):
                pop = pg.population(prob = prob, size = kwargs['search_pop_size'], seed = cpt+1)
                pop = algo.evolve(pop)


                """ It seems pop.get_f() is already sorted, no need to perform the following sorting """
                # if(self.problem.DO==2):
                #   front = pg.non_dominated_front_2d(pop.get_f())
                # else:
                #   ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())
                #   front = ndf[0]
                # fs = pop.get_f()[front]
                # xs = pop.get_x()[front]
                # bestidx = pg.select_best_N_mo(points = fs, N = kwargs['search_more_samples'])
                # xss = xs[bestidx]
                # fss = fs[bestidx]
                # # print('bestidx',bestidx)

                firstn = min(int(kwargs['search_more_samples']),np.shape(pop.get_f())[0])
                fss = pop.get_f()[0:firstn]
                xss = pop.get_x()[0:firstn]
                # print('firstn',firstn,int(kwargs['search_more_samples']),np.shape(pop.get_f()),xss)


                if(np.max(fss)< self.options['search_bigval']):
                    cond = True
                    bestX.append(xss)
                    break
                cpt += 1
        if (kwargs['verbose']):
            print(tid, 'OK' if cond else 'KO'); sys.stdout.flush()
            print("bestX",bestX)
        return (tid, bestX)

##### Simple constrained MOO

class SurrogateProblemCMO(object):

    def __init__(self, problem, computer, data, models, options, tid):   # data is in the normalized space, IOrig and POrig are then generated in the original space

        self.problem = problem
        self.computer = computer
        self.data = data
        self.models = models
        self.options = options

        self.tid = tid

        self.D     = self.data.D[tid]
        self.IOrig = self.problem.IS.inverse_transform(np.array(self.data.I[tid], ndmin=2))[0]
        self.POrig = self.problem.PS.inverse_transform(np.array(self.data.P[tid], ndmin=2))

    def get_nobj(self):
        if(self.options['search_algo']=='pso' or self.options['search_algo']=='cmaes'):
            return 1
        else:
            return self.problem.DO

    def get_bounds(self):

        DP = self.problem.DP

        return ([0. for i in range(DP)], [1. for  i in range(DP)])

    # Acquisition function
    def af(self, x):

        out_of_range = False
        for o in range(self.problem.DO):
            (lower_bound, upper_bound) = self.problem.OS.bounds[o]
            (mu, var) = self.models[o].predict(x, tid=self.tid)
            mu = mu[0][0]
            var = max(1e-18, var[0][0])
            std = np.sqrt(var)

            print ("mu: ", mu, "var: ", var)
            print ("lower_bound: ", lower_bound, "upper_bound: ", upper_bound)

            #if mu+std < lower_bound or mu-std > upper_bound:
            #    out_of_range = True
            if mu < lower_bound or mu > upper_bound:
                out_of_range = True

        if out_of_range == True:
            return [0]
        else:
            ret = 1
            for o in range(self.problem.DO):
                (mu, var) = self.models[o].predict(x, tid=self.tid)
                ret *= abs(1.0/mu[0][0])
                print ("ret: ", ret)
            return [-1.0*ret]

            #""" Expected Improvement """
            #AF=[]
            #for o in range(self.problem.DO):
            #    ymin = self.data.O[self.tid][:,o].min()
            #    (mu, var) = self.models[o].predict(x, tid=self.tid)
            #    mu = mu[0][0]
            #    var = max(1e-18, var[0][0])
            #    std = np.sqrt(var)
            #    chi = (ymin - mu-self.options['search_ei_alpha']) / std
            #    Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
            #    phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
            #    AF.append(-((ymin - mu-self.options['search_ei_alpha']) * Phi + std * phi))

            ##print ("AF: ", AF)

            #if(self.options['search_algo']=='pso' or self.options['search_algo']=='cmaes'):
            #    EI_prod = np.prod(AF)
            #    return [-1.0*EI_prod if EI_prod>0 else EI_prod]
            #else:
            #    return AF

    def fitness(self, x):   # x is the normalized space
        xi0 = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
        xi=xi0[0]

        if (any(xx==xi for xx in self.POrig)):
            cond = False
        else:
            point0 = self.D
            point2 = {self.problem.IS[k].name: self.IOrig[k] for k in range(self.problem.DI)}
            point  = {self.problem.PS[k].name: xi[k] for k in range(self.problem.DP)}
            point.update(point0)
            point.update(point2)
            # print("point", point)
            cond = self.computer.evaluate_constraints(self.problem, point)

        if (cond):
            xNorm = self.problem.PS.transform(xi0)[0]
            if(self.problem.models is not None):
                if(self.problem.driverabspath is not None):
                    modulename = Path(self.problem.driverabspath).stem  # get the driver name excluding all directories and extensions
                    sys.path.append(self.problem.driverabspath) # add path to sys
                    module = importlib.import_module(modulename) # import driver name as a module
                else:
                    raise Exception('performance models require passing driverabspath to GPTune')
                # modeldata= self.problem.models(point)
                modeldata= module.models(point)
                xNorm = np.hstack((xNorm,modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
                # print(xNorm)

            # print("cond",cond,- self.af(x),'x',x,'xi',xi)
            # print ("AF: ", self.af(xNorm))
            return self.af(xNorm)
        else:
            # print("cond",cond,float("Inf"),'x',x,'xi',xi)
            if(self.problem.DO==1): # single objective optimizer
                return [self.options['search_bigval']]
            else:
                return [self.options['search_bigval']]* self.problem.DO
            
    def obj_scipy(self, x):
        return self.fitness(x)[0]

class SearchCMO(Search):

    """
    XXX: This class, together with the underlying PyGMO only works on Intel and AMD CPUs.
    The reason is that PyGMO requires the Intel 'Thread Building Block' library to compile and execute.
    """

    def search(self, data : Data, models : Collection[Model], tid : int, **kwargs) -> np.ndarray:
        import pygmo as pg

        # print ("SearchByCMO")

        kwargs = kwargs['kwargs']
        print("searcher: ", kwargs["search_class"], "algorithm: ", kwargs["search_algo"])

        prob = SurrogateProblemCMO(self.problem, self.computer, data, models, self.options, tid)

        try:
            udi = eval(f'pg.{kwargs["search_udi"]}()')
        except:
            raise Exception('Unknown user-defined-island "{kwargs["search_udi"]}"')

        if(self.problem.DO==1): # single objective optimizer
            try:
                algo = eval(f'pg.{kwargs["search_algo"]}(gen = kwargs["search_gen"])')
            except:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')
            bestX = []
            cond = False
            cpt = 0
            while (not cond and cpt < kwargs['search_max_iters']):
                archi = pg.archipelago(n = kwargs['search_threads'], prob = prob, algo = algo, udi = udi, pop_size = kwargs['search_pop_size'])
                archi.evolve(n = kwargs['search_evolve'])
                archi.wait()
                champions_f = archi.get_champions_f()
                champions_x = archi.get_champions_x()
                indexes = list(range(len(champions_f)))
                indexes.sort(key=champions_f.__getitem__)
                for idx in indexes:
                    if (champions_f[idx] < self.options['search_bigval']):
                        cond = True
                        # bestX.append(np.array(self.problem.PS.inverse_transform(np.array(champions_x[idx], ndmin=2))[0]).reshape(1, self.problem.DP))
                        bestX.append(np.array(champions_x[idx]).reshape(1, self.problem.DP))
                        break
                cpt += 1
        else:                   # multi objective
            try:
                algo = eval(f'pg.algorithm(pg.{kwargs["search_algo"]}(gen = kwargs["search_gen"]))')
            except:
                raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')
            bestX = []
            cond = False
            cpt = 0
            while (not cond and cpt < kwargs['search_max_iters']):
                pop = pg.population(prob = prob, size = kwargs['search_pop_size'], seed = cpt+1)
                pop = algo.evolve(pop)

                firstn = min(int(kwargs['search_more_samples']),np.shape(pop.get_f())[0])
                fss = pop.get_f()[0:firstn]
                xss = pop.get_x()[0:firstn]
                # print('firstn',firstn,int(kwargs['search_more_samples']),np.shape(pop.get_f()),xss)

                if(np.max(fss)< self.options['search_bigval']):
                    cond = True
                    bestX.append(xss)
                    break
                cpt += 1
        if (kwargs['verbose']):
            print(tid, 'OK' if cond else 'KO'); sys.stdout.flush()
        # print("bestX",bestX)
        return (tid, bestX)




class SearchSciPy(Search):

    def search(self, data : Data, models : Collection[Model], tid : int, **kwargs) -> np.ndarray:

        if(self.problem.DO>1):
            raise Exception("'SearchSciPy' cannot be used for multi-objective search")

        kwargs = kwargs['kwargs']

        prob = SurrogateProblem(self.problem, self.computer, data, models, self.options, tid, self.models_transfer)

        if (kwargs['verbose']):
            print ("prob: ", prob)
        bestX = []

        # set seeds for the samplers using 'search_random_seed' instead of 'sample_random_seed', as they are used to generate the intial guess for scipy optimizers
        if(kwargs['search_random_seed'] is not None): 
            seed = kwargs['search_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            kwargs['sample_random_seed'] = seed
        else: 
            kwargs['sample_random_seed'] = None

        sampler = eval(f'{kwargs["sample_class"]}()')
        check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
        tmpP = sampler.sample_parameters(problem = self.problem, n_samples = 1, I = data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
        x0 = tmpP[0][0]

        lw = [0]*self.problem.DP
        up = [1]*self.problem.DP
        bounds_constraint = sp.optimize.Bounds(lw, up)
        print("searcher: ", kwargs["search_class"], "algorithm: ", kwargs["search_algo"])
        if(kwargs["search_algo"] == 'trust-constr'):
            ret = sp.optimize.minimize(prob.fitness, x0, method='trust-constr',  jac="2-point", hess=sp.optimize.SR1(),constraints=[], options={'verbose': 1}, bounds=bounds_constraint)
        elif(kwargs["search_algo"] == 'l-bfgs-b'):        
            ret = sp.optimize.minimize(fun=prob.fitness, x0=x0, bounds=bounds_constraint, method='L-BFGS-B')
        elif(kwargs["search_algo"] == 'dual_annealing'): 
            ret = sp.optimize.dual_annealing(prob.obj_scipy, bounds=list(zip(lw, up)), seed=kwargs['search_random_seed'])
        else:
            raise Exception("GPTune only supports 'l-bfgs-b', 'dual_annealing', 'trust-constr' when 'SearchSciPy' is used")

        # print(ret,'erere')
        print('>>>>Maximal acquisition function = ',ret.fun,' attained at ',ret.x)

        

        bestX.append(np.array(ret.x).reshape(1, self.problem.DP))
        return (tid, bestX)


if __name__ == '__main__':

    def objectives(point):
        print('this is a dummy definition')
        return point

    def models(point):
        print('this is a dummy definition')
        return point
    
    def models_update(data):
        print('this is a dummy definition')
        return data

    def cst1(point):
        print('this is a dummy definition')
        return point
    def cst2(point):
        print('this is a dummy definition')
        return point
    def cst3(point):
        print('this is a dummy definition')
        return point
    def cst4(point):
        print('this is a dummy definition')
        return point
    def cst5(point):
        print('this is a dummy definition')
        return point
    def cst6(point):
        print('this is a dummy definition')
        return point
    import mpi4py    
    from mpi4py import MPI
    mpi_comm = mpi4py.MPI.Comm.Get_parent()
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    (searcher, data, models, tids, kwargs) = mpi_comm.bcast(None, root=0)
    tids_loc = tids[mpi_rank:len(tids):mpi_size]
    tmpdata = searcher.search_multitask(data, models, tids_loc, i_am_manager = False, **kwargs)
    res = mpi_comm.gather(tmpdata, root=0)
    mpi_comm.Disconnect()


