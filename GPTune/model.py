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

from problem import Problem
from computer import Computer
from data import Data

import mpi4py

import concurrent
from concurrent import futures
class Model(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer, **kwargs):

        self.problem = problem
        self.computer = computer
        self.M = None

    @abc.abstractmethod
    def train(self, data : Data, **kwargs):

        raise Exception("Abstract method")

    @abc.abstractmethod
    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        raise Exception("Abstract method")

    @abc.abstractmethod
    def predict(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        raise Exception("Abstract method")


import GPy

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
            K = GPy.kern.RBF(input_dim = len(data.P[0][0]), ARD=True, name='GPy_GP')
            if (kwargs['model_sparse']):
                self.M = GPy.models.SparseGPRegression(data.P[0], data.O[0], kernel = K, num_inducing = model_inducing)
            else:
                self.M = GPy.models.GPRegression(data.P[0], data.O[0], kernel = K)
            self.M['.*Gaussian_noise.variance'].constrain_bounded(1e-10,1e-5)

#        np.random.seed(mpi_rank)
#        num_restarts = max(1, model_n_restarts // mpi_size)


        resopt = self.M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust = True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = 'lbfgs', start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)

#        self.M.param_array[:] = allreduce_best(self.M.param_array[:], resopt)[:]
        self.M.parameters_changed()

        if(multitask): # dump the hyperparameters
            theta=[]
            var=[]
            ws=[]
            kappa=[]
            sigma=[]
            # print(self.M)
            for qq in range(model_latent):
                q = self.M.kern['.*GPy_LCM%s.rbf.lengthscale'%qq]
                theta = theta + q.values.tolist()
                q = self.M.kern['.*GPy_LCM%s.rbf.variance'%qq]
                var = var + q.values.tolist() 
                q = self.M.kern['.*GPy_LCM%s.B.W'%qq]
                ws = ws + q.values.tolist() 
                q = self.M.kern['.*GPy_LCM%s.B.kappa'%qq]
                kappa = kappa + q.values.tolist() 
            for qq in range(data.NI):
                q = self.M['.*mixed_noise.Gaussian_noise_%s.variance'%qq]
                sigma = sigma + q.values.tolist()

            if(kwargs['verbose']==True):
                print('theta:',theta)
                print('var:',var)
                print('kappa:',kappa)
                print('WS:',ws)            
                print('sigma:',sigma)



            params = theta+var+kappa+sigma+ws        
        return

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
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

class Model_LCM(Model):

    def train(self, data : Data, **kwargs):

        return self.train_mpi(data, i_am_manager = True, restart_iters=list(range(kwargs['model_restarts'])), **kwargs)

    def train_mpi(self, data : Data, i_am_manager : bool, restart_iters : Collection[int] = None, **kwargs):

        if (kwargs['RCI_mode'] is False):
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
            _ = mpi_comm.bcast((self, data, restart_iters, kwargs_tmp), root=mpi4py.MPI.ROOT)
            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()
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
                    return kern.train_kernel(X = data.P, Y = data.O, computer = self.computer, kwargs = kwargs)
                res = list(executor.map(fun, restart_iters, timeout=None, chunksize=1))

        else:
            def fun(restart_iter):
                # np.random.seed(restart_iter)
                np.random.seed()
                kern = LCM(input_dim = len(data.P[0][0]), num_outputs = data.NI, Q = Q)
                return kern.train_kernel(X = data.P, Y = data.O, computer = self.computer, kwargs = kwargs)
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
        self.M = GPy.models.GPCoregionalizedRegression(data.P, data.O, kern, likelihoods_list = likelihoods_list)

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

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    # make prediction on a single sample point of a specific task tid
    def predict(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        (mu, var) = self.M.predict_noiseless(x)   # predict_noiseless ueses precomputed Kinv and Kinv*y (generated at GPCoregionalizedRegression init, which calls inference in GPy/inference/latent_function_inference/exact_gaussian_inference.py) to compute mu and var, with O(N^2) complexity, see "class PosteriorExact(Posterior): _raw_predict" of GPy/inference/latent_function_inference/posterior.py.

        return (mu, var)

    def gen_model_from_hyperparameters(self, data : Data, hyperparameters : list, **kwargs):
        if (kwargs['RCI_mode'] is False):
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

        self.M.optimize_restarts(num_restarts = num_restarts, robust = True, verbose = self.verbose, parallel = (num_processes is not None), num_processes = num_processes, messages = "True", optimizer = 'lbfgs', start = None, max_iters = max_iters, ipython_notebook = False, clear_after_finish = True)

        # Unfix noise variance now that we have initialized the model
        for i in range(len(self.M.layers)):
            self.M.layers[i].Gaussian_noise.variance.unfix()

        self.M.optimize_restarts(num_restarts = num_restarts, robust = True, verbose = self.verbose, parallel = (num_processes is not None), num_processes = num_processes, messages = "True", optimizer = 'lbfgs', start = None, max_iters = max_iters, ipython_notebook = False, clear_after_finish = True)

    def update(self, newdata : Data, do_train: bool = False, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

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
    from mpi4py import MPI
    mpi_comm = mpi4py.MPI.Comm.Get_parent()
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    (modeler, data, restart_iters, kwargs) = mpi_comm.bcast(None, root=0)
    restart_iters_loc = restart_iters[mpi_rank:len(restart_iters):mpi_size]
    tmpdata = modeler.train_mpi(data, i_am_manager = False, restart_iters = restart_iters_loc, **kwargs)
    res = mpi_comm.gather(tmpdata, root=0)
    mpi_comm.Disconnect()

