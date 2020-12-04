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
from model import *
from search import *


def TLA2(self, Tnew, modelers, **kwargs):

    stats = {
        "time_total": 0,
        "time_search": 0,
        "time_fun": 0
    }

    t0=time.time_ns()

    Tnew = self.problem.IS.transform(Tnew)

    aprxopts = []
    O = []

    kwargs = copy.deepcopy(self.options)
    kwargs['search_acq'] = 'mean'

    searcher =  eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')

    for t in Tnew:

        newdata = Data(self.problem, I = [t], P = [], O = [], D = [{}])

        t1=time.time_ns()
        res = searcher.search(data = newdata, models = modelers, tid = 0, **kwargs)[1]
        t2=time.time_ns()
        stats["time_search"] += (t2 - t1)/1e9
        
        newdata.P = [res]
        aprxopts += newdata.P

        t1=time.time_ns()
        newdata.O = self.computer.evaluate_objective(problem = self.problem, I = newdata.I, P = newdata.P, D = newdata.D, options = kwargs)
        O += newdata.O
        t2 = time.time_ns()
        stats["time_fun"] += (t2 - t1)/1e9

    t3 = time.time_ns()
    stats["time_total"] += (t3 - t0)/1e9

    return (aprxopts, O, stats)

