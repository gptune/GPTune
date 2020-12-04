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


from typing import Collection
import math
import numpy as np
import copy
import itertools

import skopt.space
from skopt.space import *

from problem import Problem


class Categoricalnorm(Categorical):
    def transform(self, X):
        lens=len(self.categories)
        Xt = super(Categoricalnorm, self).transform(X)
        tmp1=[]
        for xt in Xt:
            if(len(xt)==1):
                tmp = min(1. - 1e-12, xt[0]+1e-12)
            else:
                ii = next(i for i,v in enumerate(xt) if v > 0)
                tmp = ii/lens+0.01
            tmp1.append(tmp)
        return tmp1

    def inverse_transform(self, Xt):
        lens=len(self.categories)
        tmp1=[]
        # print(Xt,'wideeeefd')
        for xt in Xt:
            # print(xt,'widfd')
            tmp=[0 for ii in range(lens)]
            tmp[math.floor(min(xt,1-1e-12)*lens)]=1
            tmp1.append(tmp)
        # print(tmp1,'before inverse_transform',Xt,xt*lens)

        return super(Categoricalnorm, self).inverse_transform(tmp1)

    @property
    def transformed_size(self):
        return 1


class Data(object):
    # To GPTune I is 2D numpy array. To user I is a list of lists
    # To GPTune P is a list/collection of 2D numpy array with column dimension corresponding to PS dimension. To user P is a list of (list of lists)
    # To GPTune and user O is a list/collection of 2D numpy array with column dimension 1 for single-objective function.
    # To GPTune and user D is a list/collection of dictionaries
    def __init__(self, problem : Problem, I = None, P = None, O = None, D = None):
    # def __init__(self, problem : Problem, I : np.ndarray = None, P : Collection[np.ndarray] = None, O : Collection[np.ndarray] = None):

        self.problem = problem

        # if (not self.check_inputs(I)):
        #     raise Exception("")

        self.I = I

        # if (not self.check_parameters(P)):
        #     raise Exception("")

        self.P = P

        # if (not self.check_outputs(O)):
        #     raise Exception("")

        self.O = O


        self.D = D

    @property
    def NI(self):

        if (self.I is None):
            return 0
        else:
            return len(self.I)

    def check_inputs(self, I: np.ndarray) -> bool:

        cond = True
        if (I is not None):
            if (not (I.ndim == 2 and I.shape[1] == self.problem.DI)):
                cond = False

        return cond

    def check_parameters(self, P: Collection[np.ndarray]) -> bool:

        cond = True
        if (P is not None):
            for x in P:
                if (x is not None and len(x) > 0):
                    if not (x.ndim == 2 and x.shape[1] == problem.DP):
                        cond = False
                        break

        return cond

    def check_outputs(self, O: Collection[np.ndarray]) -> bool:

        cond = True
        if (O is not None):
            for o in O:
                if (o is not None and len(o) > 0):
                    if not (o.ndim == 2 and o.shape[1] == problem.DO):
                        cond = False
                        break

        return cond

    # TODO
    def points2kwargs(self):

        # transform the self.I and self.P into a list of dictionaries

        pass

    def normalized(self):

        # Returns a copy of the normalized data (self must be originalized)

        if (self.I is None):
            I = None
        else:
            I = np.array(self.problem.IS.transform(self.I), ndmin=2)
        if (self.P is None):
            P = None
        else:
            P = [np.array(self.problem.PS.transform(x), ndmin=2) for x in self.P]
        O = copy.copy(self.O)
        dataNorm = Data(problem = self.problem, I = I, P = P, O = O)

        return dataNorm

    def originalized(self):

        # Returns a copy of the originalized data (self must be normalized)

        if (self.I is None):
            I = None
        else:
            I = np.array(self.problem.IS.inverse_transform(self.I), ndmin=2)
        if (self.P is None):
            P = None
        else:
            P = [np.array(self.problem.PS.inverse_transform(x), ndmin=2) for x in self.P]
        O = copy.copy(self.O)
        dataOrig = Data(problem = self.problem, I = I, P = P, O = O)

        return dataOrig

    # TODO
    def merge(self, newdata):

        # merge the newdata with self, making sure that the Ts coincide

        if (not np.array_equal(self.I, newdata.I)):
            raise Exception("The tasks in the newdata should be the same as the current tasks")

        if (not np.array_equal(self.D, newdata.D)):
            raise Exception("The tasks dictionaries in the newdata should be the same as the current tasks")

        self.P = [np.concatenate((self.P[i], newdata.P[i])) for i in range(len(self.P))]
        self.O = [np.concatenate((self.O[i], newdata.O[i])) for i in range(len(self.O))]

    def fusion(self, newdata):

        # similar to merge, but tasks can be different

        if (self.P is None and newdata.P is not None and len(newdata.P) > 0):
            self.P = []#[None for tid in range(len(self.I))]
        if (self.O is None and newdata.O is not None and len(newdata.O) > 0):
            self.O = []#[None for tid in range(len(self.I))]
        for i, t in enumerate(newdata.I):
            idt = np.where((self.I == t).all(axis=1))[0] if self.I is not None else []
            if (len(idt) > 0):
                idt = idt[0]
                for j, x in enumerate(newdata.P[i]):
                    idx = np.where((self.P[idt] == x).all(axis=1))[0]
                    if (len(idx) == 0):
                        if (self.P[idt] is not None):
                            self.P[idt] = np.concatenate((self.P[idt], x.reshape((1, self.problem.DP))))
                        else:
                            self.P[idt] = x.reshape((1, self.problem.DP))
                        if (newdata.O is not None):
                            y = newdata.O[i][j]
                            if (self.O[idt] is not None):
                                self.O[idt] = np.concatenate((self.O[idt], y.reshape((1, self.problem.DO))))
                            else:
                                self.O[idt] = y.reshape((1, self.problem.DO))
                    else:
                        # x already in self.P, we chose not to update it or add another entry
                        pass
            else:
                if self.I is None:
                    self.I = t.reshape((1, self.problem.DI))
                else:
                    self.I = np.concatenate((self.I, t.reshape((1, self.problem.DI))))
                self.P.append(np.array(newdata.P[i], ndmin=2))#.reshape((newdata.P[i].shape[0], self.problem.DP)))
                if (newdata.O is not None):
                    self.O.append(np.array(newdata.O[i], ndmin=2))

    def PO2XY(self):

        X = self.P[0]
        Y = self.O[0]

        return (X, Y)

    def IPO2XY(self):

        X = np.array([np.concatenate((self.I[i], self.P[i][j])) for i in range(len(self.I)) for j in range(self.P[i].shape[0])])
        Y = np.array(list(itertools.chain.from_iterable(self.O)))

        return (X, Y)

