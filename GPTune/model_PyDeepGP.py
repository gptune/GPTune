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

import deepgp

from problem import Problem
from computer import Computer
from data import Data
from model import Model


class Model_DGP(Model):

    def train(self, data : Data, **kwargs):

        self.data = data

        #--------- Data preparation ----------#

        multitask = len(data.I) > 1
        if (multitask):
            (X, Y) = data.IPO2XY()
        else:
            (X, Y) = data.PO2XY()

        #--------- Parameter selection ----------#

        # Model parameters

        model_layers = kwargs['model_layers']
        if (kwargs['model_latent'] is not None):
            Q = kwargs['model_latent']
        else:
            #Q = 2
            Q = self.problem.DI + self.problem.DP
        nDims = [Y.shape[1]] + model_layers * [Q] + [X.shape[1]]
        if (kwargs['model_inducing'] is not None):
            num_inducing = kwargs['model_inducing']
        else:
            #num_inducing = X.shape[0]
            #num_inducing = int(X.shape[0] / math.log(X.shape[0], 10) if X.shape[0] > 0 else X.shape[0])
            num_inducing = int(5 * np.sqrt(X.shape[0]))
            print('num_inducing', num_inducing)
        # Whether to use back-constraint for variational posterior
        back_constraint = False
        #back_constraint = True
        # Dimensions of the MLP back-constraint if set to true
        encoder_dims = None
        #encoder_dims = [[X.shape[0]],[X.shape[0]],[X.shape[0]]]

        # Optimization parameters

        num_restarts  = kwargs['model_restarts']
        verbose       = kwargs['verbose']
        num_processes = kwargs['model_processes']
        max_iters     = kwargs['model_max_iters']

        #--------- Model Construction ----------#

#        self.M = RegressionModel()

        kerns = [GPy.kern.RBF(input_dim=Q, ARD=True) + GPy.kern.Bias(input_dim=Q) for lev in range(model_layers)]
        kerns.append(GPy.kern.RBF(input_dim=X.shape[1], ARD=True) + GPy.kern.Bias(input_dim=X.shape[1]))
        self.M = deepgp.DeepGP\
                (
                        nDims,
                        Y,
                        X = X,
                        num_inducing = num_inducing,
                        likelihood = None,
                        inits = 'PCA',
                        name = 'deepgp',
                        kernels = kerns,
                        obs_data = 'cont',
                        back_constraint = back_constraint,
                        encoder_dims = encoder_dims,
                        mpi_comm = kwargs['mpi_comm'],
                        mpi_root = 0,
                        repeatX = False,
                        inference_method = None
                )#, **kwargs)

        #--------- Optimization ----------#

#        # Make sure initial noise variance gives a reasonable signal to noise ratio.
#        # Fix to that value for a few iterations to avoid early local minima
#
        for i in range(len(self.M.layers)):
            output_var = self.M.layers[i].Y.var() if i==0 else self.M.layers[i].Y.mean.var()
            self.M.layers[i].Gaussian_noise.variance = output_var*0.01
            self.M.layers[i].Gaussian_noise.variance.fix()

        self.M.optimize(messages = True, max_iters = 100)
#        
#        # Unfix noise variance now that we have initialized the model
#        for i in range(len(self.M.layers)):
#            self.M.layers[i].Gaussian_noise.variance.unfix()

#        self.M.obslayer.likelihood.variance[:] = Y.var()*0.01
#        for layer in self.M.layers:
#            layer.kern.rbf.variance.fix(warning=False)
#            layer.likelihood.variance.fix(warning=False)
#
#        self.M.optimize(messages = True, max_iters = 100)

        for layer in self.M.layers:
            layer.kern.rbf.variance.constrain_positive(warning=False)

        self.M.optimize(messages = True, max_iters = 100)

        for layer in self.M.layers:
            layer.likelihood.variance.constrain_positive(warning=False)

        #self.M.optimize(messages = True, max_iters = max_iters)
        self.M.optimize_restarts\
                (
                        num_restarts = num_restarts,
                        robust = True,
                        verbose = verbose,
                        parallel = (num_processes is not None and num_processes > 1),
                        num_processes = num_processes,
                        messages = True,
                        optimizer = 'lbfgs',
                        start = None,
                        max_iters = max_iters,
                        ipython_notebook = False,
                        clear_after_finish = True
                )

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        if (do_train):
            self.train(newdata, **kwargs)
        else:
            (X, Y) = newdata.IPO2XY()
            self.M.append_XY(Y, X)

    def predict(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        multitask = len(self.data.I) > 1
        if (multitask):
            (mu, var) = self.M.predict(np.concatenate((self.data.I[tid], points)).reshape((1, self.problem.DI + self.problem.DP)))
        else:
            (mu, var) = self.M.predict(points.reshape((1, self.problem.DP)))

        return (mu, var)

