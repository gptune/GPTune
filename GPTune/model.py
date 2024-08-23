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
from typing import Collection, Tuple
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from problem import Problem
from computer import Computer
from data import Data
import gpflow
from gpflow.utilities import parameter_dict
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


import math

import concurrent
from concurrent import futures
class Model(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer, mf=None, **kwargs):

        self.problem = problem
        self.computer = computer
        self.mf=mf
        self.M = None
        self.M_last = None # used for TLA with model regression
        self.M_stacked = [] # used for TLA with model stacking
        self.num_samples_stacked = [] # number of samples used for models in model stacking

    def mfnorm(self,xnorm):
        return self.mf(self.problem.PS.inverse_transform(np.array(xnorm, ndmin=2))[0])

    @abc.abstractmethod
    def train(self, data : Data, **kwargs):

        raise Exception("Abstract method")

    @abc.abstractmethod
    def train_stacked(self, data : Data, num_source_tasks : int, **kwargs):

        raise Exception("Abstract method")

    @abc.abstractmethod
    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        raise Exception("Abstract method")

    @abc.abstractmethod
    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        raise Exception("Abstract method")

    @abc.abstractmethod
    def predict_last(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        raise Exception("Abstract method")



class Model_GPy_LCM(Model):
    
#model_threads=1
#model_processes=1
#model_groups=1
#model_restarts=1
#model_max_iters=15000
#model_latent=0
#model_sparse=False
#model_inducing=None
#model_layers=2

    def train(self, data : Data, **kwargs):
        import GPy
        if kwargs['model_random_seed'] != None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        import copy
        self.M_last = copy.deepcopy(self.M)

        multitask = len(data.I) > 1

        if (kwargs['model_latent'] is None):
            model_latent = data.NI
        else:
            model_latent = kwargs['model_latent']

        if (kwargs['model_sparse'] and kwargs['model_inducing'] is None):
            if (multitask):
                lenx = sum([len(P) for P in data.P])
            else:
                lenx = len(data.P)
            model_inducing = int(min(lenx, 3 * np.sqrt(lenx)))

        GPy.util.linalg.jitchol.__defaults__ = (kwargs['model_max_jitter_try'],)

        if (multitask):
            if(self.mf is not None):
                raise Exception("Model_GPy_LCM cannot yet handle prior mean functions in LCM")
                
            kernels_list = [GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True) for k in range(model_latent)]
            K = GPy.util.multioutput.LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, kernels_list = kernels_list, W_rank = 1, name='GPy_LCM')
            K['.*rbf.variance'].constrain_fixed(1.) #For this kernel, K.*.B.kappa and B.W encode the variance now.
            # print(K)
            if (kwargs['model_sparse']):
                self.M = GPy.models.SparseGPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K, num_inducing = model_inducing)
            else:
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K)
            for qq in range(model_latent):
                self.M['.*mixed_noise.Gaussian_noise_%s.variance'%qq].constrain_bounded(1e-10,1e-5)
        else:
            if kwargs['model_kern'] == 'RBF':
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif kwargs['model_kern'] == 'Exponential' or kwargs['model_kern'] == 'Matern12':
                K = GPy.kern.Exponential(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif kwargs['model_kern'] == 'Matern32':
                K = GPy.kern.Matern32(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif kwargs['model_kern'] == 'Matern52':
                K = GPy.kern.Matern52(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            else:
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            
            if(self.mf is not None):
                gpymf = GPy.core.Mapping(len(data.P[0][0]),1)         
                gpymf.f = self.mfnorm
                gpymf.update_gradients = lambda a,b: None
            else:
                gpymf = None

            if (kwargs['model_sparse']):
                self.M = GPy.models.SparseGPRegression(data.P[0], data.O[0], kernel = K, num_inducing = model_inducing, mean_function=gpymf)
            else:
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K, mean_function=gpymf)
            self.M['.*Gaussian_noise.variance'].constrain_bounded(1e-10,1e-5)

#        np.random.seed(mpi_rank)
#        num_restarts = max(1, model_n_restarts // mpi_size)

        resopt = self.M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust = True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = kwargs['model_optimizer'], start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)

#        self.M.param_array[:] = allreduce_best(self.M.param_array[:], resopt)[:]
        self.M.parameters_changed()

        # dump the hyperparameters
        if(multitask):
            hyperparameters = {
                "rbf_lengthscale": [],
                "variance": [],
                "B_W": [],
                "B_kappa": [],
                "noise_variance": []
            }
            model_stats = {
                "log_marginal_likelihood": self.M._log_marginal_likelihood
            }
            modeling_options = {}
            modeling_options["model_kernel"] = "RBF"
            if kwargs["model_sparse"] == True:
                modeling_options["model_method"] = "SparseGPCoregionalizedRegression"
                modeling_options["model_sparse"] = "yes"
            else:
                modeling_options["model_method"] = "GPCoregionalizedRegression"
                modeling_options["model_sparse"] = "no"
            modeling_options["multitask"] = "yes"

            for qq in range(model_latent):
                q = self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq]
                hyperparameters["rbf_lengthscale"].append(q.values.tolist())

                q = self.M.kern['sum.GPy_LCM%s.rbf.variance'%qq]
                hyperparameters["variance"].append(q.values.tolist())

                q = self.M.kern['sum.GPy_LCM%s.B.W'%qq]
                hyperparameters["B_W"].append(q.values.tolist())

                q = self.M.kern['sum.GPy_LCM%s.B.kappa'%qq]
                hyperparameters["B_kappa"].append(q.values.tolist())

            for qq in range(data.NI):
                q = self.M['mixed_noise.Gaussian_noise_%s.variance'%qq]
                hyperparameters["noise_variance"].append(q.values.tolist())

            if(kwargs['verbose']==True):
                print("rbf_lengthscale: ", hyperparameters["rbf_lengthscale"])
                print("variance: ", hyperparameters["variance"])
                print("B_W: ", hyperparameters["B_W"])
                print("B_kappa: ", hyperparameters["B_kappa"])
                print("noise_variance: ", hyperparameters["noise_variance"])

        else:
            hyperparameters = {
                "lengthscale": [],
                "variance": [],
                "noise_variance": []
            }
            model_stats = {
                "log_marginal_likelihood": self.M._log_marginal_likelihood
            }
            modeling_options = {}
            modeling_options["model_kern"] = kwargs["model_kern"]
            if kwargs["model_sparse"] == True:
                modeling_options["model_method"] = "SparseGPRegression"
                modeling_options["model_sparse"] = "yes"
            else:
                modeling_options["model_method"] = "GPRegression"
                modeling_options["model_sparse"] = "no"
            modeling_options["multitask"] = "no"

            q = self.M.kern["GPy_GP.lengthscale"]
            hyperparameters["lengthscale"] = hyperparameters["lengthscale"] + q.values.tolist()

            q = self.M.kern["GPy_GP.variance"]
            hyperparameters["variance"] = hyperparameters["variance"] + q.values.tolist()

            q = self.M["Gaussian_noise.variance"]
            hyperparameters["noise_variance"] = hyperparameters["noise_variance"] + q.values.tolist()

            if(kwargs['verbose']==True):
                print("lengthscale: ", hyperparameters["lengthscale"])
                print("variance: ", hyperparameters["variance"])
                print("noise_variance: ", hyperparameters["noise_variance"])
            print ("modeler: ", kwargs['model_class'])
            print ("M: ", self.M)

        return (hyperparameters, modeling_options, model_stats)

    def train_stacked(self, data : Data, num_source_tasks, **kwargs):

        # note: model stacking works only for single task tuning
        # each source task model is a single-task model, and target model is also a single-task model

        self.train(data, **kwargs)

        if len(self.M_stacked) < 1+num_source_tasks:
            self.M_stacked.append(self.M)
            self.num_samples_stacked.append(len(data.P[0]))
        elif len(self.M_stacked) == 1+num_source_tasks: # residual for the current target task
            self.M_stacked[num_source_tasks] = self.M
            self.num_samples_stacked[num_source_tasks] = len(data.P[0])
        else:
            print ("Unexpected. Stacking model count does not match")

        return self.M_stacked

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        if len(self.M_stacked) > 0: # stacked model
            x = np.empty((1, points.shape[0] + 1))
            x[0,:-1] = points
            x[0,-1] = tid

            (mu, var) = self.M_stacked[0].predict_noiseless(x)
            var[0][0] = max(1e-18, var[0][0])
            num_samples_prior = self.num_samples_stacked[0]

            for i in range(1, len(self.M_stacked), 1):
                (mu_, var_) = self.M_stacked[i].predict_noiseless(x)
                var_[0][0] = max(1e-18, var_[0][0])
                num_samples_current = self.num_samples_stacked[i]
                alpha = 1.0 # relative importance of the prior and current ones
                beta = float((alpha*num_samples_current)/(alpha*num_samples_current+num_samples_prior))
                mu[0][0] += mu_[0][0]
                var[0][0] = math.pow(var_[0][0], beta) * math.pow(var[0][0], (1.0-beta))
                num_samples_prior = num_samples_current
        else:
            if not len(points.shape) == 2:
                points = np.atleast_2d(points)
            x = np.empty((points.shape[0], points.shape[1] + 1))
            x[:,:-1] = points
            x[:,-1] = tid
            (mu, var) = self.M.predict_noiseless(x,full_cov=full_cov)

        return (mu, var)

    def predict_last(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        if self.M_last != None:
            (mu, var) = self.M_last.predict_noiseless(x)
        else:
            (mu, var) = self.M.predict_noiseless(x)

        return (mu, var)

    def get_correlation_metric(self, delta):
        print("In model.py, delta = ", delta)
        Q = delta # number of latent processes 
        B = np.zeros((delta, delta, Q))
        for i in range(Q):
            currentLCM = getattr(self.M.sum, f"GPy_LCM{i}")
            Wq = currentLCM.B.W.values
            Kappa_q = currentLCM.B.kappa.values
            B[:, :, i] = np.outer(Wq, Wq) + np.diag(Kappa_q)
            # print("In model.py, i = ", i)
            # print(B[:, :, i])
            
        # return C_{i, i'}
        C = np.zeros((delta, delta))
        for i in range(delta):
            for ip in range(i, delta):
                C[i, ip] = np.linalg.norm(B[i, ip, :]) / np.sqrt(np.linalg.norm(B[i, i, :]) * np.linalg.norm(B[ip, ip, :]))
        return C

    def gen_model_from_hyperparameters(self, data : Data, hyperparameters : dict, modeling_options : dict, **kwargs):

        if kwargs['model_random_seed'] != None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        if modeling_options["multitask"] == "yes":
            multitask = True
        else:
            multitask = False

        if modeling_options["model_sparse"] == "yes":
            model_sparse = True
        else:
            model_sparse = False

        if (kwargs['model_latent'] is None):
            model_latent = data.NI
        else:
            model_latent = kwargs['model_latent']

        if (model_sparse and kwargs['model_inducing'] is None):
            if (multitask):
                lenx = sum([len(P) for P in data.P])
            else:
                lenx = len(data.P)
            model_inducing = int(min(lenx, 3 * np.sqrt(lenx)))

        GPy.util.linalg.jitchol.__defaults__ = (kwargs['model_max_jitter_try'],)

        if (multitask):
            kernels_list = [GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True) for k in range(model_latent)]
            K = GPy.util.multioutput.LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, kernels_list = kernels_list, W_rank = 1, name='GPy_LCM')
            K['.*rbf.variance'].constrain_fixed(1.) #For this kernel, K.*.B.kappa and B.W encode the variance now.
            # print(K)
            if modeling_options["model_sparse"] == "SparseGPCoregionalizedRegression":
                self.M = GPy.models.SparseGPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K, num_inducing = model_inducing)
            elif modeling_options["model_method"] == "GPCoregionalizedRegression":
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K)
            else:
                print ("unsupported modeling method: ", modeling_options['model_method'], " will use GPCoregionalizedRegression")
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K)

                for qq in range(model_latent):
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][0] = hyperparameters["rbf_lengthscale"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][1] = hyperparameters["rbf_lengthscale"][qq][1]
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][2] = hyperparameters["rbf_lengthscale"][qq][2]
                    self.M.kern['sum.GPy_LCM%s.rbf.variance'%qq] = hyperparameters["variance"][qq]
                    self.M.kern['sum.GPy_LCM%s.B.W'%qq][0][0] = hyperparameters["B_W"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.B.W'%qq][1][0] = hyperparameters["B_W"][qq][1]
                    self.M.kern['sum.GPy_LCM%s.B.kappa'%qq][0] = hyperparameters["B_kappa"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.B.kappa'%qq][1] = hyperparameters["B_kappa"][qq][1]
                    self.M.kern['mixed_noise.Gaussian_noise_%s.variance'%qq] = hyperparameters["noise_variance"][qq][0]

                for qq in range(model_latent):
                    self.M['.*mixed_noise.Gaussian_noise_%s.variance'%qq].constrain_bounded(1e-10,1e-5)

                self.M.parameters_changed()

                print ("reproduced model: ", self.M)

        else:
            if modeling_options['model_kern'] == 'RBF':
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Exponential' or modeling_options['model_kern'] == 'Matern12':
                K = GPy.kern.Exponential(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Matern32':
                K = GPy.kern.Matern32(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Matern52':
                K = GPy.kern.Matern52(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            else:
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            if modeling_options['model_method'] == 'SparseGPRegression':
                self.M = GPy.models.SparseGPRegression(data.P[0], data.O[0], kernel = K, num_inducing = model_inducing)
            elif modeling_options['model_method'] == 'GPRegression':
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K)
            else:
                print ("unsupported modeling method: ", modeling_options['model_method'], " will use GPRegression")
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K)
            self.M['.*Gaussian_noise.variance'].constrain_bounded(1e-10,1e-5)

            print ("len: ", len(self.M.kern['GPy_GP.lengthscale']))

            for i in range(len(self.M.kern['GPy_GP.lengthscale'])):
                self.M.kern['GPy_GP.lengthscale'][i] = hyperparameters["lengthscale"][i]

            self.M.kern['GPy_GP.variance'] = hyperparameters["variance"][0]
            self.M['Gaussian_noise.variance'] = hyperparameters["noise_variance"][0]

            self.M.parameters_changed()

            print ("reproduced model: ", self.M)

        return


class Model_GPFlow_LCM(Model):

#model_threads=1
#model_processes=1
#model_groups=1
#model_restarts=1
#model_max_iters=15000
#model_latent=0
#model_sparse=False
#model_inducing=None
#model_layers=2


    def bounded_parameter(self,low, high, default):
        """Make noise tfp Parameter with optimization bounds. (From Hengrui Luo)"""
        #affine = tfb.AffineScalar(shift=tf.cast(low, tf.float64),
        #                          scale=tf.cast(high-low, tf.float64))
        affine_scale = tfb.Scale(scale=tf.cast(high-low, tf.float64))
        affine_shift = tfb.Shift(shift=tf.cast(low, tf.float64))
        sigmoid = tfb.Sigmoid()
        logistic = tfb.Chain([affine_shift, affine_scale, sigmoid])
        parameter = gpflow.Parameter(default, transform=logistic, dtype=tf.float64)
        return parameter

    def bounded_parameter_sig(self,low, high, default, n_tuple=1):
        """Make lengthscale tfp Parameter with optimization bounds. (From Hengrui Luo)"""
        sigmoid = tfb.Sigmoid(tf.cast(low, tf.float64), tf.cast(high, tf.float64))
        if n_tuple>1:
            parameter = [gpflow.Parameter(default, transform=sigmoid, dtype=tf.float64) for i in range(n_tuple)]
            parameter = tuple(parameter)
        else:
            parameter = gpflow.Parameter(default, transform=sigmoid, dtype=tf.float64)
        return parameter

    def contains_coregion_kernel(self,kernel):
        """
        Recursively checks if a kernel or any of its sub-kernels is a Coregion kernel.
        """
        if isinstance(kernel, gpflow.kernels.Coregion):
            return True
        elif hasattr(kernel, 'kernels'):  # Check if it's a combination kernel
            return any(self.contains_coregion_kernel(sub_kernel) for sub_kernel in kernel.kernels)
        else:
            return False



    def train(self, data : Data, **kwargs):
        if kwargs['model_random_seed'] != None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        import copy
        self.M_last = copy.deepcopy(self.M)

        multitask = len(data.I) > 1

        if (kwargs['model_latent'] is None):
            model_latent = data.NI
        else:
            model_latent = kwargs['model_latent']

        if (kwargs['model_sparse'] and kwargs['model_inducing'] is None):
            if (multitask):
                lenx = sum([len(P) for P in data.P])
            else:
                lenx = len(data.P)
            model_inducing = int(min(lenx, 3 * np.sqrt(lenx)))

        # GPy.util.linalg.jitchol.__defaults__ = (kwargs['model_max_jitter_try'],)

        if (multitask):
            if(self.mf is not None):
                raise Exception("Model_GPFlow_LCM cannot yet handle prior mean functions in LCM")

            if (kwargs['model_sparse']):
                raise Exception("model_sparse not supported in Model_GPFlow_LCM")
            else:

                datapad_P = copy.deepcopy(data.P)
                datapad_O = copy.deepcopy(data.O)
                for i in range(data.NI):
                    tid = i
                    datapad_P[i] = np.hstack((datapad_P[i], tid*np.ones((datapad_P[i].shape[0], 1))))
                    datapad_O[i] = np.hstack((datapad_O[i], tid*np.ones((datapad_O[i].shape[0], 1))))

                datapad_P = np.vstack(datapad_P)
                datapad_O = np.vstack(datapad_O)

                # Lists to store the individual kernels and likelihoods
                kernels = []
                likelihoods = []

                # Loop to create the kernels and likelihoods
                for i in range(model_latent):

                    input_dim = len(data.P[0][0])
                    if kwargs['model_kern'] == 'RBF':
                        k = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, active_dims=list(range(input_dim)), variance=1.0, name='GPFlow_GP')
                    elif kwargs['model_kern'] == 'Exponential' or kwargs['model_kern'] == 'Matern12':
                        k = gpflow.kernels.Matern12(lengthscales=[1.0] * input_dim, active_dims=list(range(input_dim)), variance=1.0, name='GPFlow_GP')
                    elif kwargs['model_kern'] == 'Matern32':
                        k = gpflow.kernels.Matern32(lengthscales=[1.0] * input_dim, active_dims=list(range(input_dim)), variance=1.0, name='GPFlow_GP')
                    elif kwargs['model_kern'] == 'Matern52':
                        k = gpflow.kernels.Matern52(lengthscales=[1.0] * input_dim, active_dims=list(range(input_dim)), variance=1.0, name='GPFlow_GP')
                    else:
                        k = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')

                    gpflow.set_trainable(k.variance, False) #K.*.B.kappa and B.W encode the variance now.
                    coreg = gpflow.kernels.Coregion(output_dim=data.NI, rank=1, active_dims=[len(data.P[0][0])])
                    
                    # Multiply Matern32 and Coregion kernels
                    kern = k * coreg
                    kernels.append(kern)

                for i in range(data.NI):    
                    # Create a Gaussian likelihood for each kernel
                    noise_variance = self.bounded_parameter(1e-10, 1e-5, 1e-5/2)
                    likelihood = gpflow.likelihoods.Gaussian(variance=noise_variance)
                    likelihoods.append(gpflow.likelihoods.Gaussian())

                # Sum the kernels to create the final kernel
                kern = kernels[0]
                for i in range(model_latent-1):
                    kern += kernels[i+1]

                # Create the SwitchedLikelihood with the list of Gaussian likelihoods
                lik = gpflow.likelihoods.SwitchedLikelihood(likelihoods)

                # Now build the GP model
                self.M = gpflow.models.VGP((datapad_P, datapad_O), kernel=kern, likelihood=lik)

                for qq in range(model_latent):
                    self.M.likelihood.likelihoods[i].variance.assign(tf.clip_by_value(self.M.likelihood.likelihoods[i].variance, 1e-10, 1e-5))

        else:
            input_dim = len(data.P[0][0])

            if kwargs['model_kern'] == 'RBF':
                K = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')
            elif kwargs['model_kern'] == 'Exponential' or kwargs['model_kern'] == 'Matern12':
                K = gpflow.kernels.Matern12(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')
            elif kwargs['model_kern'] == 'Matern32':
                K = gpflow.kernels.Matern32(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')
            elif kwargs['model_kern'] == 'Matern52':
                K = gpflow.kernels.Matern52(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')
            else:
                K = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, variance=1.0, name='GPFlow_GP')
                        
            if self.mf is not None:
                def gpflow_mf(X):
                    xnorm = X.numpy()  # Convert TensorFlow tensor to NumPy array
                    transformed_x = self.problem.PS.inverse_transform(np.array(xnorm, ndmin=2))
                    return tf.convert_to_tensor(self.mf(transformed_x[0]), dtype=tf.float64)
                
                gpflow_mf = gpflow.mean_functions.Lambda(gpflow_mf)
            else:
                gpflow_mf = None

            noise_variance = self.bounded_parameter(1e-10, 1e-5, 1e-5/2)
            likelihood = gpflow.likelihoods.Gaussian(variance=noise_variance)
            
            if (kwargs['model_sparse']):
                inducing_points = data.P[0][:model_inducing]
                self.M = gpflow.models.SGPR(data=(data.P[0], data.O[0]), kernel=K, inducing_variable=inducing_points, likelihood=likelihood, mean_function=gpflow_mf)
            else:
                self.M = gpflow.models.GPR(data=(data.P[0], data.O[0]), kernel=K, likelihood=likelihood, mean_function=gpflow_mf)            
            
#        np.random.seed(mpi_rank)
#        num_restarts = max(1, model_n_restarts // mpi_size)

        # q1 = self.M.kern["GPy_GP.lengthscale"]
        # q2 = self.M.kern["GPy_GP.variance"]
        # q3 = self.M["Gaussian_noise.variance"]
        # print("p0",q1.values.tolist(),q2.values.tolist(),q3.values.tolist(),self.M._log_marginal_likelihood)

        # resopt = self.M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust = True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = kwargs['model_optimizer'], start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
        # iteration = resopt[0].funct_eval
        
        # opt_options = {
        #     'maxiter': kwargs['model_max_iters'],
        #     'gtol': 1e-05,
        #     'eps': 1e-08,
        #     'disp': kwargs['verbose'],
        #     'maxcor': 10,
        #     'ftol': 1e-32,
        #     'maxfun': 1000,
        #     'iprint': -1,
        #     'maxls': 100
        # }


        opt_options = {
            'maxiter': kwargs['model_max_iters'],
            'disp': kwargs['verbose'],
        }        
        
        opt = gpflow.optimizers.Scipy()
        resopt = opt.minimize(self.M.training_loss, self.M.trainable_variables, options=opt_options,method="L-BFGS-B")
        
        # dump the hyperparameters
        if(multitask):
            hyperparameters = {
                "lengthscale": [],
                "variance": [],
                "B_W": [],
                "B_kappa": [],
                "noise_variance": []
            }
            model_stats = {
                "log_marginal_likelihood": -self.M.training_loss().numpy()
            }
            modeling_options = {}
            modeling_options["model_kernel"] = kwargs["model_kern"]
            if kwargs["model_sparse"] == True:
                modeling_options["model_method"] = "SparseGPCoregionalizedRegression"
                modeling_options["model_sparse"] = "yes"
            else:
                modeling_options["model_method"] = "GPCoregionalizedRegression"
                modeling_options["model_sparse"] = "no"
            modeling_options["multitask"] = "yes"

            params = parameter_dict(self.M)

            for qq in range(model_latent):

                q = np.atleast_1d(params['.kernel.kernels[%s].kernels[0].lengthscales'%qq].numpy())    
                hyperparameters["lengthscale"].append(q.tolist())

                q = np.atleast_1d(params['.kernel.kernels[%s].kernels[0].variance'%qq].numpy())    
                hyperparameters["variance"].append(q.tolist())

                q = np.atleast_1d(params['.kernel.kernels[%s].kernels[1].W'%qq].numpy()) 
                hyperparameters["B_W"].append(q.tolist())

                q = np.atleast_1d(params['.kernel.kernels[%s].kernels[1].kappa'%qq].numpy()) 
                hyperparameters["B_kappa"].append(q.tolist())

            for qq in range(data.NI):
                q = np.atleast_1d(params['.likelihood.likelihoods[%s].variance'%qq].numpy())
                hyperparameters["noise_variance"].append(q.tolist())

            if(kwargs['verbose']==True):
                print("lengthscale: ", hyperparameters["lengthscale"])
                print("variance: ", hyperparameters["variance"])
                print("B_W: ", hyperparameters["B_W"])
                print("B_kappa: ", hyperparameters["B_kappa"])
                print("noise_variance: ", hyperparameters["noise_variance"])
        else:
            
            hyperparameters = {
                "lengthscale": [],
                "variance": [],
                "noise_variance": []
            }

            model_stats = {
                "log_marginal_likelihood": -self.M.training_loss().numpy()
            }

            modeling_options = {}
            modeling_options["model_kern"] = kwargs["model_kern"]
            if kwargs["model_sparse"] == True:
                modeling_options["model_method"] = "SparseGPRegression"
                modeling_options["model_sparse"] = "yes"
            else:
                modeling_options["model_method"] = "GPRegression"
                modeling_options["model_sparse"] = "no"
            modeling_options["multitask"] = "no"

            params = parameter_dict(self.M)
            q = np.atleast_1d(params[".kernel.lengthscales"].numpy())
            hyperparameters["lengthscale"] = hyperparameters["lengthscale"] + q.tolist()
            q = np.atleast_1d(params[".kernel.variance"].numpy())
            hyperparameters["variance"] = hyperparameters["variance"] + q.tolist()

            q = np.atleast_1d(params[".likelihood.variance"].numpy())
            hyperparameters["noise_variance"] = hyperparameters["noise_variance"] + q.tolist()

            if(kwargs['verbose']==True):
                print("lengthscale: ", hyperparameters["lengthscale"])
                print("variance: ", hyperparameters["variance"])
                print("noise_variance: ", hyperparameters["noise_variance"])
            print ("modeler: ", kwargs['model_class'])
            print ("M: ", self.M)

        print(model_stats)
        return (hyperparameters, modeling_options, model_stats)

    def train_stacked(self, data : Data, num_source_tasks, **kwargs):

        # note: model stacking works only for single task tuning
        # each source task model is a single-task model, and target model is also a single-task model

        self.train(data, **kwargs)

        if len(self.M_stacked) < 1+num_source_tasks:
            self.M_stacked.append(self.M)
            self.num_samples_stacked.append(len(data.P[0]))
        elif len(self.M_stacked) == 1+num_source_tasks: # residual for the current target task
            self.M_stacked[num_source_tasks] = self.M
            self.num_samples_stacked[num_source_tasks] = len(data.P[0])
        else:
            print ("Unexpected. Stacking model count does not match")

        return self.M_stacked

    def update(self, newdata : Data, do_train: bool = False, **kwargs):
        
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        if len(self.M_stacked) > 0: # stacked model
            if(self.contains_coregion_kernel(self.M_stacked[0].kernel)):
                x = np.empty((1, points.shape[0] + 1))
                x[0,:-1] = points
                x[0,-1] = tid
            else:
                x = points

            (mu, var) = self.M_stacked[0].predict_f(x)
            var[0][0] = max(1e-18, var[0][0])
            num_samples_prior = self.num_samples_stacked[0]

            for i in range(1, len(self.M_stacked), 1):
                (mu_, var_) = self.M_stacked[i].predict_f(x)
                var_[0][0] = max(1e-18, var_[0][0])
                num_samples_current = self.num_samples_stacked[i]
                alpha = 1.0 # relative importance of the prior and current ones
                beta = float((alpha*num_samples_current)/(alpha*num_samples_current+num_samples_prior))
                mu[0][0] += mu_[0][0]
                var[0][0] = math.pow(var_[0][0], beta) * math.pow(var[0][0], (1.0-beta))
                num_samples_prior = num_samples_current
        else:
            if not len(points.shape) == 2:
                points = np.atleast_2d(points)
            
            # print(self.M)
            if(self.contains_coregion_kernel(self.M.kernel)):
                x = np.empty((points.shape[0], points.shape[1] + 1))
                x[:,:-1] = points
                x[:,-1] = tid
            else:
                x = points
            (mu, var) = self.M.predict_f(x,full_cov=full_cov)
            # print(mu, var, 'gpy')
        return (mu, var)

    def predict_last(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        if self.M_last != None:
            (mu, var) = self.M_last.predict_noiseless(x)
        else:
            (mu, var) = self.M.predict_noiseless(x)

        return (mu, var)

    def get_correlation_metric(self, delta):
        print("In model.py, delta = ", delta)
        Q = delta # number of latent processes 
        B = np.zeros((delta, delta, Q))
        for i in range(Q):
            currentLCM = getattr(self.M.sum, f"GPy_LCM{i}")
            Wq = currentLCM.B.W.values
            Kappa_q = currentLCM.B.kappa.values
            B[:, :, i] = np.outer(Wq, Wq) + np.diag(Kappa_q)
            # print("In model.py, i = ", i)
            # print(B[:, :, i])
            
        # return C_{i, i'}
        C = np.zeros((delta, delta))
        for i in range(delta):
            for ip in range(i, delta):
                C[i, ip] = np.linalg.norm(B[i, ip, :]) / np.sqrt(np.linalg.norm(B[i, i, :]) * np.linalg.norm(B[ip, ip, :]))
        return C

    def gen_model_from_hyperparameters(self, data : Data, hyperparameters : dict, modeling_options : dict, **kwargs):

        if kwargs['model_random_seed'] != None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        if modeling_options["multitask"] == "yes":
            multitask = True
        else:
            multitask = False

        if modeling_options["model_sparse"] == "yes":
            model_sparse = True
        else:
            model_sparse = False

        if (kwargs['model_latent'] is None):
            model_latent = data.NI
        else:
            model_latent = kwargs['model_latent']

        if (model_sparse and kwargs['model_inducing'] is None):
            if (multitask):
                lenx = sum([len(P) for P in data.P])
            else:
                lenx = len(data.P)
            model_inducing = int(min(lenx, 3 * np.sqrt(lenx)))

        GPy.util.linalg.jitchol.__defaults__ = (kwargs['model_max_jitter_try'],)

        if (multitask):
            kernels_list = [GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True) for k in range(model_latent)]
            K = GPy.util.multioutput.LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, kernels_list = kernels_list, W_rank = 1, name='GPy_LCM')
            K['.*rbf.variance'].constrain_fixed(1.) #For this kernel, K.*.B.kappa and B.W encode the variance now.
            # print(K)
            if modeling_options["model_sparse"] == "SparseGPCoregionalizedRegression":
                self.M = GPy.models.SparseGPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K, num_inducing = model_inducing)
            elif modeling_options["model_method"] == "GPCoregionalizedRegression":
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K)
            else:
                print ("unsupported modeling method: ", modeling_options['model_method'], " will use GPCoregionalizedRegression")
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.P, Y_list = data.O, kernel = K)

                for qq in range(model_latent):
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][0] = hyperparameters["rbf_lengthscale"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][1] = hyperparameters["rbf_lengthscale"][qq][1]
                    self.M.kern['sum.GPy_LCM%s.rbf.lengthscale'%qq][2] = hyperparameters["rbf_lengthscale"][qq][2]
                    self.M.kern['sum.GPy_LCM%s.rbf.variance'%qq] = hyperparameters["variance"][qq]
                    self.M.kern['sum.GPy_LCM%s.B.W'%qq][0][0] = hyperparameters["B_W"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.B.W'%qq][1][0] = hyperparameters["B_W"][qq][1]
                    self.M.kern['sum.GPy_LCM%s.B.kappa'%qq][0] = hyperparameters["B_kappa"][qq][0]
                    self.M.kern['sum.GPy_LCM%s.B.kappa'%qq][1] = hyperparameters["B_kappa"][qq][1]
                    self.M.kern['mixed_noise.Gaussian_noise_%s.variance'%qq] = hyperparameters["noise_variance"][qq][0]

                for qq in range(model_latent):
                    self.M['.*mixed_noise.Gaussian_noise_%s.variance'%qq].constrain_bounded(1e-10,1e-5)

                self.M.parameters_changed()

                print ("reproduced model: ", self.M)

        else:
            if modeling_options['model_kern'] == 'RBF':
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Exponential' or modeling_options['model_kern'] == 'Matern12':
                K = GPy.kern.Exponential(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Matern32':
                K = GPy.kern.Matern32(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            elif modeling_options['model_kern'] == 'Matern52':
                K = GPy.kern.Matern52(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            else:
                K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            if modeling_options['model_method'] == 'SparseGPRegression':
                self.M = GPy.models.SparseGPRegression(data.P[0], data.O[0], kernel = K, num_inducing = model_inducing)
            elif modeling_options['model_method'] == 'GPRegression':
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K)
            else:
                print ("unsupported modeling method: ", modeling_options['model_method'], " will use GPRegression")
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K)
            self.M['.*Gaussian_noise.variance'].constrain_bounded(1e-10,1e-5)

            print ("len: ", len(self.M.kern['GPy_GP.lengthscale']))

            for i in range(len(self.M.kern['GPy_GP.lengthscale'])):
                self.M.kern['GPy_GP.lengthscale'][i] = hyperparameters["lengthscale"][i]

            self.M.kern['GPy_GP.variance'] = hyperparameters["variance"][0]
            self.M['Gaussian_noise.variance'] = hyperparameters["noise_variance"][0]

            self.M.parameters_changed()

            print ("reproduced model: ", self.M)

        return


class Model_LCM(Model):
    def train(self, data : Data, **kwargs):
        import copy
        self.M_last = copy.deepcopy(self.M)

        return self.train_mpi(data, i_am_manager = True, restart_iters=list(range(kwargs['model_restarts'])), **kwargs)

    def train_mpi(self, data : Data, i_am_manager : bool, restart_iters : Collection[int] = None, **kwargs):
        import GPy
        if (kwargs['RCI_mode']== False):
            import mpi4py
            from lcm import LCM

        if (kwargs['model_latent'] is None):
            Q = data.NI
        else:
            Q = kwargs['model_latent']

        if (kwargs['distributed_memory_parallelism'] and i_am_manager):
            mpi_comm = self.computer.spawn(__file__, nproc=kwargs['model_restart_processes'], nthreads=kwargs['model_restart_threads'], kwargs=kwargs) # XXX add args and kwargs
            kwargs_tmp = kwargs
            # print("kwargs_tmp",kwargs_tmp)

            if "mpi_comm" in kwargs_tmp:
                del kwargs_tmp["mpi_comm"]   # mpi_comm is not picklable
            
            
            import copy
            data_tmp = copy.deepcopy(data)
            # YL: substract the prior mean before calling the C modeling training function 
            mf_saved=self.mf
            if(self.mf is not None):
                for i in range(len(data.P)):
                    for p in range(data.P[i].shape[0]):
                        data_tmp.O[i][p,0]=data_tmp.O[i][p,0]-self.mfnorm(data.P[i][p,:])            
            self.mf = None
            _ = mpi_comm.bcast((self, data_tmp, restart_iters, kwargs_tmp), root=mpi4py.MPI.ROOT)
            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()
            self.mf = mf_saved
            res=[]
            for p in range(int(kwargs['model_restart_processes'])):
                res = res + tmpdata[p]

        elif (kwargs['shared_memory_parallelism']): #YL: not tested

            #with concurrent.futures.ProcessPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers = kwargs['model_restart_threads']) as executor:
                def fun(restart_iter):
                    # if ('seed' in kwargs):
                    #     seed = kwargs['seed'] * kwargs['model_restart_threads'] + restart_iter
                    # else:
                    #     seed = restart_iter
                    # np.random.seed(seed)
                    ## np.random.seed()
                    kern = LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, Q = Q)
                    # if (restart_iter == 0 and self.M is not None):
                    #     kern.set_param_array(self.M.kern.get_param_array())
                    
                    import copy
                    data_O = copy.deepcopy(data.O)
                    # YL: substract the prior mean before calling the C modeling training function 
                    if(self.mf is not None):
                        for i in range(len(data.P)):
                            for p in range(data.P[i].shape[0]):
                                data_O[i][p,0]=data_O[i][p,0]-self.mfnorm(data.P[i][p,:])
                    return kern.train_kernel(X = data.P, Y = data_O, computer = self.computer, kwargs = kwargs)
                res = list(executor.map(fun, restart_iters, timeout=None, chunksize=1))

        else:
            def fun(restart_iter):
                # np.random.seed(restart_iter)
                if kwargs['model_random_seed'] == None:
                    np.random.seed()
                else:
                    seed = kwargs['model_random_seed']
                    if data.P is not None:
                        for P_ in data.P:
                            seed += len(P_)
                    np.random.seed(seed)
                kern = LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, Q = Q)
                import copy
                data_O = copy.deepcopy(data.O)
                # YL: substract the prior mean before calling the C modeling training function 
                if(self.mf is not None):
                    for i in range(len(data.P)):
                        for p in range(data.P[i].shape[0]):
                            data_O[i][p,0]=data_O[i][p,0]-self.mfnorm(data.P[i][p,:])
                return kern.train_kernel(X = data.P, Y = data_O, computer = self.computer, kwargs = kwargs)
            res = list(map(fun, restart_iters))

        if (kwargs['distributed_memory_parallelism'] and i_am_manager == False):
            return res

        kern = LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, Q = Q)
        best_result = min(res, key = lambda x: x[1])
        bestxopt = best_result[0]
        neg_log_marginal_likelihood = best_result[1]
        gradients = best_result[2]
        iteration = best_result[3]
        kern.set_param_array(bestxopt)
        if(kwargs['verbose']==True):
            # print('hyperparameters:', kern.get_param_array())
            print('theta:',kern.theta)
            print('var:',kern.var)
            print('kappa:',kern.kappa)
            print('sigma:',kern.sigma)
            print('WS:',kern.WS)




        # YL: likelihoods needs to be provided, since K operator doesn't take into account sigma/jittering, but Kinv does. The GPCoregionalizedRegression intialization will call inference in GPy/interence/latent_function_inference/exact_gaussian_inference.py, and add to diagonals of the K operator with sigma+1e-8
        likelihoods_list = [GPy.likelihoods.Gaussian(variance = kern.sigma[i], name = "Gaussian_noise_%s" %i) for i in range(data.NI)]
        import copy
        data_O = copy.deepcopy(data.O)
        # YL: GPCoregionalizedRegression initialization in GPy (unlike GPRegression) doesn't accept mean_function, so we need to subtract mean from data.O for calling the prediction function later. Also, we need to add back the mean in the predict function below 
        if(self.mf is not None):
            for i in range(len(data.P)):
                for p in range(data.P[i].shape[0]):
                    data_O[i][p,0]=data_O[i][p,0]-self.mfnorm(data.P[i][p,:])
        self.M = GPy.models.GPCoregionalizedRegression(data.P, data_O, kern, likelihoods_list = likelihoods_list)

        #print ("kernel: " + str(kern))
        #print ("bestxopt:" + str(bestxopt))
        #print ("neg_log_marginal_likelihood:" + str(neg_log_marginal_likelihood))
        #print ("gradients: " + str(gradients))
        #print ("iteration: " + str(iteration))
        #for i in range(data.NI):
        #    print ("i: " + str(i))
        #    print ("sigma: " + str(kern.sigma[i]))
        #    print ("likelihoods_list: " + str(likelihoods_list[i].to_dict()))
        #print ("likelihoods_list_len: " + str(data.NI))
        #print ("self.M: " + str(self.M))

        return (bestxopt, neg_log_marginal_likelihood, gradients, iteration)

    def train_stacked(self, data : Data, num_source_tasks, **kwargs):

        if len(self.M_stacked) < 1+num_source_tasks:
            self.M_stacked.append(self.M)
            self.num_samples_stacked.append(len(data.P[0]))
        elif len(self.M_stacked) == 1+num_source_tasks: # residual for the current target task
            self.M_stacked[num_source_tasks] = self.M
            self.num_samples_stacked[num_source_tasks] = len(data.P[0])
        else:
            print ("Unexpected. Stacking model count does not match")

        return self.M_stacked

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    # make prediction on a single sample point of a specific task tid
    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        if len(self.M_stacked) > 0: # stacked model
            x = np.empty((1, points.shape[0] + 1))
            x[0,:-1] = points
            x[0,-1] = tid

            (mu, var) = self.M_stacked[0].predict_noiseless(x)
            var[0][0] = max(1e-18, var[0][0])
            num_samples_prior = self.num_samples_stacked[0]

            for i in range(1, len(self.M_stacked), 1):
                (mu_, var_) = self.M_stacked[i].predict_noiseless(x)
                if(self.mf is not None):
                    mu_[0][0] = mu_[0][0] + self.mfnorm(x)
                var_[0][0] = max(1e-18, var_[0][0])
                num_samples_current = self.num_samples_stacked[i]
                alpha = 1.0 # relative importance of the prior and current ones
                beta = float((alpha*num_samples_current)/(alpha*num_samples_current+num_samples_prior))
                mu[0][0] += mu_[0][0]
                var[0][0] = math.pow(var_[0][0], beta) * math.pow(var[0][0], (1.0-beta))
                num_samples_prior = num_samples_current
        else:
            if not len(points.shape) == 2:
                points = np.atleast_2d(points)
            x = np.empty((points.shape[0], points.shape[1] + 1))
            x[:,:-1] = points
            x[:,-1] = tid
            (mu, var) = self.M.predict_noiseless(x,full_cov=full_cov) # predict_noiseless ueses precomputed Kinv and Kinv*y (generated at GPCoregionalizedRegression init, which calls inference in GPy/inference/latent_function_inference/exact_gaussian_inference.py) to compute mu and var, with O(N^2) complexity, see "class PosteriorExact(Posterior): _raw_predict" of GPy/inference/latent_function_inference/posterior.py.
            if(self.mf is not None):
                for i in range(points.shape[0]):
                    mu[i] = mu[i] + self.mfnorm(x[i,:])

        return (mu, var)

    # make prediction on a single sample point of a specific task tid
    def predict_last(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        if self.M_last != None:
            (mu, var) = self.M_last.predict_noiseless(x)   # predict_noiseless ueses precomputed Kinv and Kinv*y (generated at GPCoregionalizedRegression init, which calls inference in GPy/inference/latent_function_inference/exact_gaussian_inference.py) to compute mu and var, with O(N^2) complexity, see "class PosteriorExact(Posterior): _raw_predict" of GPy/inference/latent_function_inference/posterior.py.
        else:
            (mu, var) = self.M.predict_noiseless(x)   # predict_noiseless ueses precomputed Kinv and Kinv*y (generated at GPCoregionalizedRegression init, which calls inference in GPy/inference/latent_function_inference/exact_gaussian_inference.py) to compute mu and var, with O(N^2) complexity, see "class PosteriorExact(Posterior): _raw_predict" of GPy/inference/latent_function_inference/posterior.py.

        return (mu, var)

    def gen_model_from_hyperparameters(self, data : Data, hyperparameters : list, **kwargs):
        if (kwargs['RCI_mode']== False):
            from lcm import LCM

        if (kwargs['model_latent'] is None):
            Q = data.NI
        else:
            Q = kwargs['model_latent']

        kern = LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, Q = Q)
        #print ("received hyperparameters: " + str(hyperparameters))
        kern.set_param_array(hyperparameters)

        likelihoods_list = [GPy.likelihoods.Gaussian(variance = kern.sigma[i], name = "Gaussian_noise_%s" %i) for i in range(data.NI)]
        self.M = GPy.models.GPCoregionalizedRegression(data.P, data.O, kern, likelihoods_list = likelihoods_list)

        return

class Model_DGP(Model):

    def train(self, data : Data, **kwargs):

        multitask = len(self.I) > 1

        if (multitask):
            X = np.array([np.concatenate((self.I[i], self.P[i][j])) for i in range(len(self.I)) for j in range(self.P[i].shape[0])])
        else:
            X = self.P[0]
        Y = np.array(list(itertools.chain.from_iterable(self.O)))

        #--------- Model Construction ----------#
        model_n_layers = 2
        # Define what kernels to use per layer
        kerns = [GPy.kern.RBF(input_dim=Q, ARD=True) + GPy.kern.Bias(input_dim=Q) for lev in range(model_n_layers)]
        kerns.append(GPy.kern.RBF(input_dim=X.shape[1], ARD=True) + GPy.kern.Bias(input_dim=X.shape[1]))
        # Number of inducing points to use
        if (num_inducing is None):
            if (multitask):
                lenx = sum([len(X) for X in self.P])
            else:
                lenx = len(self.P)
#            num_inducing = int(min(lenx, 3 * np.sqrt(lenx)))
            num_inducing = lenx
        # Whether to use back-constraint for variational posterior
        back_constraint = False
        # Dimensions of the MLP back-constraint if set to true
        encoder_dims=[[X.shape[0]],[X.shape[0]],[X.shape[0]]]

        nDims = [Y.shape[1]] + model_n_layers * [Q] + [X.shape[1]]
#        self.M = deepgp.DeepGP(nDims, Y, X=X, num_inducing=num_inducing, likelihood = None, inits='PCA', name='deepgp', kernels=kerns, obs_data='cont', back_constraint=True, encoder_dims=encoder_dims, mpi_comm=mpi_comm, self.mpi_root=0, repeatX=False, inference_method=None)#, **kwargs):
        self.M = deepgp.DeepGP(nDims, Y, X=X, num_inducing=num_inducing, likelihood = None, inits='PCA', name='deepgp', kernels=kerns, obs_data='cont', back_constraint=False, encoder_dims=None, mpi_comm=None, mpi_root=0, repeatX=False, inference_method=None)#, **kwargs):
#        self.M = deepgp.DeepGP([Y.shape[1], Q, Q, X.shape[1]], Y, X=X, kernels=[kern1, kern2, kern3], num_inducing=num_inducing, back_constraint=back_constraint)

        #--------- Optimization ----------#
        # Make sure initial noise variance gives a reasonable signal to noise ratio.
        # Fix to that value for a few iterations to avoid early local minima
        for i in range(len(self.M.layers)):
            output_var = self.M.layers[i].Y.var() if i==0 else self.M.layers[i].Y.mean.var()
            self.M.layers[i].Gaussian_noise.variance = output_var*0.01
            self.M.layers[i].Gaussian_noise.variance.fix()

        self.M.optimize_restarts(num_restarts = num_restarts, robust = True, verbose = self.verbose, parallel = (num_processes is not None), num_processes = num_processes, messages = "True", optimizer = kwargs['model_optimizer'], start = None, max_iters = max_iters, ipython_notebook = False, clear_after_finish = True)

        # Unfix noise variance now that we have initialized the model
        for i in range(len(self.M.layers)):
            self.M.layers[i].Gaussian_noise.variance.unfix()

        self.M.optimize_restarts(num_restarts = num_restarts, robust = True, verbose = self.verbose, parallel = (num_processes is not None), num_processes = num_processes, messages = "True", optimizer = kwargs['model_optimizer'], start = None, max_iters = max_iters, ipython_notebook = False, clear_after_finish = True)

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        (mu, var) = self.M.predict(np.concatenate((self.I[tid], x)).reshape((1, self.DT + self.DI)))

        return (mu, var)


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
    import mpi4py                  
    from mpi4py import MPI
    mpi_comm = mpi4py.MPI.Comm.Get_parent()
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    (modeler, data, restart_iters, kwargs) = mpi_comm.bcast(None, root=0)
    restart_iters_loc = restart_iters[mpi_rank:len(restart_iters):mpi_size]
    tmpdata = modeler.train_mpi(data, i_am_manager = False, restart_iters = restart_iters_loc, **kwargs)
    res = mpi_comm.gather(tmpdata, root=0)
    mpi_comm.Disconnect()

