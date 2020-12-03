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


class GPTune_MB(object):

    def __init__(self, tp : TuningProblem, computer : Computer = None, options : Options = None, **kwargs):

        """
        tp: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
        computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
        options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
        """

        smax = int(np.floor(np.log10(options['budget_max']/options['budget_min'])/np.log10(options['budget_base'])))
        self.budgets=[options['budget_max']/options['budget_base']**x for x in range(smax+1)]
        # print(self.budgets)

        parameter_space = tp.parameter_space
        output_space = tp.output_space
        objectives = tp.objective
        constraints = tp.constraints

        """ insert "budget" as the first dimension of the input space """
        inputs = [Real     (options['budget_min']-1e-12, options['budget_max'], transform="normalize", name="budget")]

        for n,p in zip(tp.input_space.dimension_names,tp.input_space.dimensions):
            if (isinstance(p, Real)):
                inputs.append(Real(p.bounds[0], p.bounds[1], transform="normalize", name=n))
            elif (isinstance(p, Integer)):
                inputs.append(Integer(p.bounds[0], p.bounds[1], transform="normalize", name=n))
            elif (isinstance(p, Categorical)):
                inputs.append(Categoricalnorm (list(p.bounds), transform="onehot", name=n))
            else:
                raise Exception("Unknown parameter type")

        # print(inputs)
        input_space = Space(inputs)

        self.tp = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)
        self.computer = computer
        self.options  = options
        self.data     = Data(tp)


    def MB_LCM(self, NS = None, Igiven = None, **kwargs):
        """
        Igiven       : a list of tasks
        NS           : number of samples in the highest budget arm
        """

        np.set_printoptions(suppress=False,precision=4)
        print('\n\n\n------Starting MB_LCM (multi-arm bandit with LCM) with %d samples for task'%(NS),Igiven)

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

        self.NSs=[int(self.options['budget_max']/x*NS) for x in self.budgets]
        info = [[x,y] for x,y in zip(self.budgets,self.NSs)]
        print('total samples:',info)

        data = Data(self.tp)   # having the budgets not fully sampled before SH
        data1 = Data(self.tp)  # having the budgets fully sampled before SH
        data1.I=[]
        data1.P=[]
        data1.O=[]
        data1.D=[]

        for s in range(len(self.budgets)): # loop over the budget levels
            budget = self.budgets[s]
            ns = self.NSs[s]
            newtasks=[]
            for s1 in range(s,len(self.budgets)):
                for t in range(len(Igiven)):
                    budget1 = self.budgets[s1]
                    tmp = [budget1]+Igiven[t]
                    newtasks.append(tmp)

            gt = GPTune(self.tp, computer=self.computer, data=data, options=self.options)
            (data, modeler, stats0) = gt.MLA(NS=ns, Igiven=newtasks, NI=len(newtasks), NS1=int(ns/2))
            data1.I += data.I[0:len(Igiven)]
            data1.P += data.P[0:len(Igiven)]
            data1.O += data.O[0:len(Igiven)]
            data1.D += data.D[0:len(Igiven)]
            del data.I[0:len(Igiven)]
            del data.P[0:len(Igiven)]
            del data.O[0:len(Igiven)]
            del data.D[0:len(Igiven)]


            stats['time_total'] += stats0['time_total']
            stats['time_fun'] += stats0['time_fun']
            stats['time_model'] += stats0['time_model']
            stats['time_search'] += stats0['time_search']
            stats['time_sample_init'] += stats0['time_sample_init']

        # print(data1.I)
        # print(data1.P)
        # print(data1.O)
        self.data.I = Igiven
        self.data.P = data1.P[0:len(Igiven)]  # this will be updated by SH
        self.data.O = data1.O[0:len(Igiven)]  # this will be updated by SH
        #todo SH on each arm and return all samples of the highest fidelity in self.data

        return (copy.deepcopy(self.data), stats)

