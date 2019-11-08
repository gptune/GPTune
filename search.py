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


import mpi4py
from mpi4py import MPI
from mpi4py import futures

from problem import Problem
from computer import Computer
from data import Data
from model import Model

class Search(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer):
        self.problem = problem
        self.computer = computer

    @abc.abstractmethod
    def search(self, data : Data, model : Model, tid : int, **kwargs) -> np.ndarray:

        raise Exception("Abstract method")

    def search_multitask(self, data : Data, model : Model, tids : Collection[int] = None, i_am_manager : bool = True, **kwargs) -> Collection[np.ndarray]:

        if (tids is None):
            tids = list(range(data.NI))

        if (kwargs['distributed_memory_parallelism'] and i_am_manager):
            mpi_comm = self.computer.spawn(__file__, kwargs['search_multitask_processes'], kwargs['search_multitask_threads'], kwargs=kwargs) # XXX add args and kwargs
            kwargs_tmp = kwargs
            if "mpi_comm" in kwargs_tmp:
                del kwargs_tmp["mpi_comm"]   # mpi_comm is not picklable
            _ = mpi_comm.bcast((self, data, model, tids, kwargs_tmp), root=mpi4py.MPI.ROOT)
            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()
            res=[]
            for p in range(int(kwargs['search_multitask_processes'])):
                res = res + tmpdata[p]


        elif (kwargs['shared_memory_parallelism']):
            
            #with concurrent.futures.ProcessPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
                # fun = functools.partial(self.search, data = data, model = model, kwargs = kwargs)
                # res = list(executor.map(fun, tids, timeout=None, chunksize=1))

                def fun(tid):
                    return self.search(data=data,model = model, tid =tid, kwargs = kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize=1))								
        else:
            fun = functools.partial(self.search, data, model, kwargs = kwargs)
            res = list(map(fun, tids))
        # print(res)
        res.sort(key = lambda x : x[0])		
        return res

class SurrogateProblem(object):

    def __init__(self, problem, computer, data, model, tid):

        self.problem = problem
        self.computer = computer
        self.data = data
        self.model = model

        self.tid = tid

        self.t     = self.data.T[tid]
        self.XOrig = self.data.X[tid]

    def get_bounds(self):

        DP = self.problem.DP

        return ([0. for i in range(DP)], [1. for  i in range(DP)])

    # Acquisition function
    def ei(self, x):

        """ Expected Improvement """

        ymin = self.data.Y[self.tid].min()
        (mu, var) = self.model.predict(x, tid=self.tid)
        mu = mu[0][0]
        var = max(1e-18, var[0][0])
        std = np.sqrt(var)
        chi = (ymin - mu) / std
        Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
        phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
        EI = (ymin - mu) * Phi + var * phi

        return EI

    def fitness(self, x):

        xi = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
        if (any(np.array_equal(xx, xi) for xx in self.XOrig)):
            cond = False
        else:
            point2 = {self.problem.IS[k].name: self.t[k] for k in range(self.problem.DI)}
            point  = {self.problem.PS[k].name: x[k] for k in range(self.problem.DP)}
            point.update(point2)
            cond = self.computer.evaluate_constraints(self.problem, point)
        if (cond):
            return (- self.ei(x),)
        else:
            return (float("Inf"),)

import pygmo as pg

class SearchPyGMO(Search):

    """
    XXX: This class, together with the underlying PyGMO only works on Intel-based CPUs.
    The reason is that PyGMO requires the Intel 'Thread Building Block' library to compile and execute.
    """

    def search(self, data : Data, model : Model, tid : int, **kwargs) -> np.ndarray:

        kwargs = kwargs['kwargs']

        prob = SurrogateProblem(self.problem, self.computer, data, model, tid)

        try:
            algo = eval(f'pg.{kwargs["search_algo"]}(gen = kwargs["search_gen"])')
        except:
            raise Exception(f'Unknown optimization algorithm "{kwargs["search_algo"]}"')

        try:
            udi = eval(f'pg.{kwargs["search_udi"]}()')
        except:
            raise Exception('Unknown user-defined-island "{kwargs["search_udi"]}"')

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
                if (champions_f[idx] < float('Inf')):
                    cond = True
                    bestX.append(np.array(self.problem.PS.inverse_transform(np.array(champions_x[idx], ndmin=2))[0]).reshape(1, self.problem.DP))
                    break
            cpt += 1

        if (kwargs['verbose']):
            print(tid, 'OK' if cond else 'KO'); sys.stdout.flush()

        return (tid, bestX)

if __name__ == '__main__':

	def objective(point):
		return point
		
	mpi_comm = MPI.Comm.Get_parent()
	mpi_rank = mpi_comm.Get_rank()
	mpi_size = mpi_comm.Get_size()
	(searcher, data, model, tids, kwargs) = mpi_comm.bcast(None, root=0)
	tids_loc = tids[mpi_rank:len(tids):mpi_size]
	tmpdata = searcher.search_multitask(data, model, tids_loc, i_am_manager = False, **kwargs)
	res = mpi_comm.gather(tmpdata, root=0) 
	mpi_comm.Disconnect()	
				
	