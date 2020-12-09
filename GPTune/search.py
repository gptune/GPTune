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


import abc
from typing import Collection
import numpy as np
import scipy as sp
import functools
from joblib import *
import sys
from sys import platform as _platform

import concurrent
from concurrent import futures
import mpi4py
from mpi4py import MPI
from mpi4py import futures

from problem import Problem
from computer import Computer
from data import Data
from sample import *
from sample_LHSMDU import *
from sample_OpenTURNS import *
from model import Model
from model_GPy import *
from model_cLCM import *
from model_PyDeepGP import *
from model_sghmc_dgp import *


class Search(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer):
        self.problem = problem
        self.computer = computer

    @abc.abstractmethod
    def search(self, data : Data, models : Collection[Model], tid : int, sampler : Sample = None, **kwargs) -> np.ndarray:

        raise Exception("Abstract method")

    def search_multitask(self, data : Data, models : Collection[Model], tids : Collection[int] = None, i_am_manager : bool = True, **kwargs) -> Collection[np.ndarray]:

        if (tids is None):
            tids = list(range(data.NI))

        if ((kwargs['distributed_memory_parallelism'] or _platform == "darwin") and i_am_manager):   # the pgymo install on mac os seems buggy if search is not spawned 
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
                    return self.search(data=data,models = models, tid =tid, **kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize=1))
        else:
            fun = functools.partial(self.search, data, models, **kwargs)
            res = list(map(fun, tids))
        # print(res)
        res.sort(key = lambda x : x[0])
        return res

