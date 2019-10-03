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

import numpy as np
from problem import Problem
from typing import Collection

class HistoricData(Data):

    def __init__(self, data, computer, options):

        self.data     = data
        self.computer = computer
        self.options  = options

#   @staticmethod
#   def load():
#
#    pass
#
#   def save(self):
#
#    pass

class Data(object):

    def __init__(self, problem : Problem, T : np.ndarray = None, X : Collection[np.ndarray] = None, Y : Collection[np.ndarray] = None):

        self.problem = problem

        if (not self.check_inputs(T)):
            raise Exception("")

        self.T = T

        if (not self.check_parameters(X)):
            raise Exception("")

        self.X = X

        if (not self.check_outputs(Y)):
            raise Exception("")

        self.Y = Y

    @property
    def NI(self):

        if (self.T is None):
            return 0
        else:
            return len(self.T)

    def check_inputs(self, T: np.ndarray) -> bool:

        cond = True
        if (T is not None):
            if (not (T.ndim == 2 and T.shape[1] == self.problem.DI)):
                cond = False

        return cond

    def check_parameters(self, X: Collection[np.ndarray]) -> bool:

        cond = True
        if (X is not None):
            for x in X:
                if (x is not None and len(x) > 0):
                    if not (x.ndim == 2 and x.shape[1] == problem.DP):
                        cond = False
                        break

        return cond

    def check_outputs(self, Y: Collection[np.ndarray]) -> bool:

        cond = True
        if (Y is not None):
            for y in Y:
                if (y is not None and len(y) > 0):
                    if not (y.ndim == 2 and y.shape[1] == problem.DO):
                        cond = False
                        break

        return cond

    # TODO
    def points2kwargs(self):

        # transform the self.T and self.X into a list of dictionaries

        pass

    # TODO
    def merge(self, newdata):

        # merge the newdata with self, making sure that the Ts coincide

        if (not np.array_equal(self.T, newdata.T)):
            raise Exception("The tasks in the newdata should be the same as the current tasks")

        self.X = [np.concatenate((self.X[i], newdata.X[i])) for i in range(len(self.X))]
        self.Y = [np.concatenate((self.Y[i], newdata.Y[i])) for i in range(len(self.Y))]

#    def insert(T = None: np.ndarray, X = None : Collection[np.ndarray], Y = None : Collection[np.ndarray]):
#
#        if (T is not None):
#            if (T.ndim == 1):
#                assert(T.shape[0] == self.problem.DI)
#            elif (T.ndim == 2):
#                assert(T.shape[1] == self.problem.DI)
#            else:
#                raise Exception("")
#            self.T.append(T)

