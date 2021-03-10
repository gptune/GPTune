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


from typing import Collection, Tuple
import numpy as np

import concurrent
from concurrent import futures
import mpi4py
from mpi4py import MPI
from mpi4py import futures

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys, os
sys.path.insert(0, os.path.abspath(__file__ + "../../../../sghmc_dgp"))
from models import RegressionModel
import pickle

from problem import Problem
from computer import Computer
from data import Data
from model import Model


class Model_SGHMC_DGP(Model):

    def train(self, data : Data, **kwargs):

        self.data = data
        (X, Y) = data.IPO2XY()

#        pickle.dump((X,Y), open('dataset.pkl', 'wb'))
        self.M = RegressionModel()
        self.M.ARGS.iterations = kwargs['model_max_iters']
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', sum(len(p) for p in self.data.P))
        self.M.ARGS.num_inducing = sum(len(p) for p in self.data.P)
        import time
        t1 = time.time()
        self.M.fit(X, Y)
        t2 = time.time()
        print('Train time: ', (t2-t1))
#        print(self.M.predict([[1., 1, 0.    , .128 ,0.064]]))

#        saver = tf.train.Saver()
#        X_placeholder = self.M.model.X_placeholder
#        pickle.dump([list(p.values()) for p in self.M.model.posterior_samples], open('aaa', 'wb'))
#        y_mean = tf.identity(self.M.model.y_mean, name="y_mean")
#        y_var = tf.identity(self.M.model.y_var, name="y_var")
#        init_op = tf.global_variables_initializer()
#        self.M.model.session.run(init_op)
#        #m, v = self.M.model.session.run((y_mean, y_var), feed_dict=feed_dict)
#        saver.save(self.M.model.session, 'mydgpmodel')

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

#        if (do_train):
            self.data.fusion(newdata)
            self.train(self.data, **kwargs)
#        else:
#            (X, Y) = newdata.IPO2XY()
#            X = np.concatenate((self.M.X, X))
#            Y = np.concatenate((self.M.Y, Y))
#            self.M.reset(X, Y)

    def predict(self, points : Collection[np.ndarray], I = None, tid : int = None, **kwargs) -> Collection[Tuple[float, float]]:

        multitask = len(self.data.I) > 1
        if (multitask):
#            X = np.concatenate((self.data.I[tid], points)).reshape((1, self.problem.DI + self.problem.DP))
            if (tid is not None):
                I = np.array(self.data.I[tid], ndmin=2)
            else:
                I = np.array(I, ndmin=2)
            points = np.array(points, ndmin=2)
            Inew = np.broadcast_to(I, (points.shape[0], I.shape[1] if len(I.shape) > 1 else 1))
            X = np.concatenate((Inew, points), axis=1)
        else:
            X = points.reshape((1, self.problem.DP))

        mu, var = self.M.predict(X)

        return (mu, var)

