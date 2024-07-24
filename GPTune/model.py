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
import copy
from typing import Collection, Tuple
import numpy as np
from problem import Problem
from computer import Computer
from data import Data

import scipy.optimize as op
import emcee
from scipy.stats import truncnorm, gamma, invgamma


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

        # q1 = self.M.kern["GPy_GP.lengthscale"]
        # q2 = self.M.kern["GPy_GP.variance"]
        # q3 = self.M["Gaussian_noise.variance"]
        # print("p0",q1.values.tolist(),q2.values.tolist(),q3.values.tolist(),self.M._log_marginal_likelihood)

        resopt = self.M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust = True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = kwargs['model_optimizer'], start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
        iteration = resopt[0].funct_eval

        # print('jiba',resopt[0],dir(resopt[0]),resopt[0].funct_eval)
#        self.M.param_array[:] = allreduce_best(self.M.param_array[:], resopt)[:]
        self.M.parameters_changed()

        # q1 = self.M.kern["GPy_GP.lengthscale"]
        # q2 = self.M.kern["GPy_GP.variance"]
        # q3 = self.M["Gaussian_noise.variance"]
        # print("popt",q1.values.tolist(),q2.values.tolist(),q3.values.tolist(),self.M._log_marginal_likelihood)


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

        return (hyperparameters, modeling_options, model_stats, iteration)

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
            print('bestxopt:', bestxopt)
            print('hyperparameters:', kern.get_param_array())
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

class Model_George(Model):
    y = []

    def gelman_rubin(self, samples):
        """
        Compute the Gelman-Rubin diagnostic statistic (R-hat) for convergence.
        
        Parameters:
        samples (np.ndarray): MCMC samples of shape (nsteps, nwalkers, ndim)
        
        Returns:
        np.ndarray: Gelman-Rubin statistic for each dimension
        """
        nsteps, nwalkers, ndim = samples.shape
        
        # Calculate the within-chain variance for each dimension
        within_chain_var = np.var(samples, axis=0, ddof=1)
        
        # Calculate the mean of the samples for each step and dimension
        chain_means = np.mean(samples, axis=0)
        
        # Calculate the mean of the means for each dimension
        mean_of_means = np.mean(chain_means, axis=0)
        
        # Calculate the between-chain variance for each dimension
        between_chain_var = np.mean((chain_means - mean_of_means) ** 2, axis=0)
        
        # Calculate the mean within-chain variance for each dimension
        mean_within_chain_var = np.mean(within_chain_var, axis=0)
        
        # Calculate the variance estimate
        var_estimate = ((nsteps - 1) / nsteps) * mean_within_chain_var + (1 / nsteps) * between_chain_var
        
        # Calculate the Gelman-Rubin statistic
        gelman_rubin_stat = np.sqrt(var_estimate / mean_within_chain_var)
        
        return gelman_rubin_stat

    def run_mcmc_with_convergence(self, sampler, initial_state, n_steps, check_interval=100, r_hat_threshold=1.01):
        nwalkers, ndim = initial_state.shape
        samples = np.zeros((n_steps, nwalkers, ndim))
        
        for i in range(0, n_steps, check_interval):
            sampler.run_mcmc(initial_state, check_interval, progress=True)
            initial_state = sampler.get_last_sample().coords
            
            current_samples = sampler.get_chain(discard=100, thin=1, flat=False)
            
            # Print shapes for debugging
            print(f"Iteration {i}: current_samples shape = {current_samples.shape}")
            
            end_index = i + check_interval
            if end_index > n_steps:
                end_index = n_steps
                
            # Check if the slice is valid
            if current_samples.shape[0] < (end_index - i):
                print(f"Warning: Not enough samples to fill the required slice. Current samples shape: {current_samples.shape}")
                continue
            
            samples[i:end_index, :, :] = current_samples[-(end_index - i):, :, :]
            
            if i >= check_interval:
                r_hat = self.gelman_rubin(samples[:i+check_interval, : , :])
                print(f"Step {i + check_interval}: R-hat = {r_hat}")
                if np.all(r_hat < r_hat_threshold):
                    print("Chains have converged.")
                    return samples[:i+check_interval, :, :]
        
        print("Reached maximum steps without full convergence.")
        return samples


    
    def log_posterior(self, params):
        log_likelihood = -self.nll(params)
        log_prior = 0
        
        length_scale = np.exp(params[2:])
        signal_variance = np.exp(params[1])
        noise_variance = np.exp(params[0])
        
        a, b = (np.log(0.1), np.log(10))
        log_prior_length_scale = np.sum(truncnorm.logpdf(np.log(length_scale), (a - np.mean([a, b])) / np.std([a, b]), (b - np.mean([a, b])) / np.std([a, b])))
        log_prior_signal_variance = invgamma.logpdf(signal_variance, a=1, scale=0.01)
        log_prior_noise_variance = invgamma.logpdf(noise_variance, a=1, scale=0.001)
        
        log_prior = log_prior_length_scale + log_prior_signal_variance + log_prior_noise_variance
        
        return log_likelihood + log_prior

    def nll(self, params):
        self.M.set_parameter_vector(params)
        return -self.M.log_likelihood(np.ravel(self.y), quiet=True)

    def grad_nll(self, params):
        self.M.set_parameter_vector(params)
        g = self.M.grad_log_likelihood(np.ravel(self.y), quiet=True)
        # print('grad ',-g[2],-g[1],-g[0])
        return -g

    def extract_hyperparameters(self, model, kernel_type):
        params = model.get_parameter_vector()
        if kernel_type == 'RBF' or kernel_type == 'Matern32' or kernel_type == 'Matern52':
        
            log_noisevariance = params[0]
            log_amplitude_squared = params[1]
            log_lengthscales = params[2:]

            noisevariance = np.exp(log_noisevariance)
            amplitude = np.exp(log_amplitude_squared ) * model.kernel.ndim
            lengthscales = np.exp(log_lengthscales)
            # print("amplitude: ", amplitude, "lengthscale ", lengthscales)
        else:
            raise Exception("TODO: IMPLEMENT OTHER KERNEL THAN RBF")

        return noisevariance, amplitude, lengthscales

    def train(self, data, **kwargs):
        import george
        if 'model_random_seed' in kwargs and kwargs['model_random_seed'] is not None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        self.M_last = copy.deepcopy(self.M)

        multitask = len(data.I) > 1 

        if multitask:
            raise Exception("TODO: IMPLEMENT MULTITASK TRAIN")
        else:
            x = data.P[0]
            self.y = data.O[0]
            input_dim = len(x[0])
            # set initial guess
            intialguess=[5e-6, 1] + [1]*input_dim
            # intialguess=[np.power(10,np.random.randn(1)), np.power(10,np.random.randn(1))] + [np.power(10,np.random.randn(1))]*input_dim

            if kwargs['model_kern'] == 'RBF':
                K = george.kernels.ExpSquaredKernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude 
            elif kwargs['model_kern'] == 'Matern32':
                K = george.kernels.Matern32Kernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude   
            elif kwargs['model_kern'] == 'Matern52':
                K = george.kernels.Matern52Kernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude             
            else:
                raise Exception("TODO: IMPLEMENT OTHER KERNELS")
  
            if kwargs['model_lowrank'] == True:
                kwargs_variable = {
                    'min_size': kwargs['model_hodlrleaf'],
                    'tol': kwargs['model_hodlrtol'],
                    'seed': 42
                }
                self.M = george.GP(kernel=K, white_noise=np.log(intialguess[0]), fit_white_noise=True, solver=george.solvers.HODLRSolver,**kwargs_variable)
            else:
                self.M = george.GP(kernel=K, white_noise=np.log(intialguess[0]), fit_white_noise=True, solver=george.solvers.BasicSolver)


        self.M.compute(x)
        
        p0 = self.M.get_parameter_vector()
        # p0[0]=1
        # p0[1]=1
        if (kwargs['verbose']):
            print("Initial Log-likelihood:", self.M.log_likelihood(np.ravel(self.y)),p0)
        noise_variance, amplitude, lengthscale = self.extract_hyperparameters(self.M,kwargs['model_kern'])
        # print(noise_variance, amplitude, lengthscale)
        # exit(1)
        
        
        bounds = [(-15, -10)] + [(None, None)] + [(-23, 19)] * input_dim
        
        
        if kwargs['mcmc']:
                # Initialize MCMC walkers around the initial guess
                ndim = len(p0)
                nwalkers = kwargs.get('nwalkers', 32)
                initial_state = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
                
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
                n_steps = kwargs.get('n_steps', 1000)
                samples = self.run_mcmc_with_convergence(sampler, initial_state, n_steps)
                
                # Get the best parameter estimate
                best_params = np.median(samples[:, -1, :], axis=0)
                resopt = type('Result', (object,), {'x': best_params, 'success': True, 'message': 'MCMC converged', 'fun': -self.log_posterior(best_params), 'nfev': samples.shape[1] * samples.shape[0]})()
        else:
            if kwargs['model_grad'] == True:
                resopt = op.minimize(self.nll, p0, jac=self.grad_nll, method="L-BFGS-B", bounds=bounds, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-32, 'gtol': 1e-05, 'eps': 1e-08, 'finite_diff_rel_step': 1e-08, 'maxfun': 1000, 'maxiter': 1000, 'iprint': -1, 'maxls': 100})
            else:
                # use finite difference, jac could be None, '2-point', '3-point', or 'cs'
                resopt = op.minimize(self.nll, p0, jac='2-point', method="L-BFGS-B", bounds=bounds, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-32, 'gtol': 1e-05, 'eps': 1e-08, 'finite_diff_rel_step': 1e-08, 'maxfun': 1000, 'maxiter': 1000, 'iprint': -1, 'maxls': 100})
        

        

        self.M.set_parameter_vector(resopt.x)
        if (kwargs['verbose']):
            print("Updated Log-likelihood:", self.M.log_likelihood(np.ravel(self.y)))

        if (kwargs['verbose']):
            print('fun      : ', resopt.fun)
            #print('hess_inv : ', sol.hess_inv)
            #print('jac      : ', jac)
            print('message  : ', resopt.message)
            print('nfev     : ', resopt.nfev)
            print('nit      : ', resopt.nit)
            print('status   : ', resopt.status)
            print('success  : ', resopt.success)
        iteration = resopt.nfev
        # Dump the hyperparameters
        if multitask:
            raise Exception("TODO: IMPLEMENT MULTITASK TRAIN")
        else:
            hyperparameters = {
                "lengthscale": [],
                "variance": [],
                "noise_variance": []
            }
            model_stats = {
                "log_marginal_likelihood": self.M.log_likelihood(np.ravel(self.y))
            }
            modeling_options = {}
            modeling_options["model_kern"] = kwargs["model_kern"]
            if 'model_lowrank' in kwargs and kwargs["model_lowrank"]:
                modeling_options["model_lowrank"] = "yes"
            else:
                modeling_options["model_lowrank"] = "no"
            modeling_options["multitask"] = "no"

            noise_variance, amplitude, lengthscale = self.extract_hyperparameters(self.M,kwargs['model_kern'])

            hyperparameters["lengthscale"] = lengthscale.tolist()
            hyperparameters["variance"] = [amplitude]
            hyperparameters["noise_variance"] = [noise_variance]

            if kwargs.get('verbose', True):
                print("lengthscale:", hyperparameters["lengthscale"])
                print("variance:", hyperparameters["variance"])
                print("noise_variance:", hyperparameters["noise_variance"])
            print("modeler:", kwargs['model_class'])
            # print("M:", self.M)

        return (hyperparameters, modeling_options, model_stats,iteration)

    def train_stacked(self, data : Data, num_source_tasks, **kwargs):

        raise Exception("TODO- TRAIN STACKED")

    def update(self, newdata : Data, do_train: bool = False, **kwargs):
        
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarray], tid : int, full_cov : bool=False, **kwargs) -> Collection[Tuple[float, float]]:

        if len(self.M_stacked) > 0: # stacked model
            raise Exception("TO DO - Implment stacked predict function")
        else:
            if not len(points.shape) == 2:
                points = np.atleast_2d(points)
            x = points
            # x = np.empty((points.shape[0], points.shape[1] + 1))
            # x[:,:-1] = points
            #  x[:,-1] = tid TODO: Add multitask later
            mu, var = self.M.predict(np.ravel(self.y), x, return_var=not full_cov)
            mu = mu[:, np.newaxis]
            var = var[:, np.newaxis]
            # print(mu, var, 'george')

            return (mu, var)

    def predict_last(self, points : Collection[np.ndarray], tid : int, **kwargs) -> Collection[Tuple[float, float]]:
        x = points
        # x = np.empty((points.shape[0], points.shape[1] + 1))
        # x[:,:-1] = points
        #  x[:,-1] = tid TODO: Add multitask later
        if self.M_last != None:
            mu, var = self.M_last.predict(np.ravel(self.y), x, return_var=not full_cov)
        else:
            mu, var = self.M.predict(np.ravel(self.y), x, return_var=not full_cov)

        mu = mu[:, np.newaxis]
        var = var[:, np.newaxis]

        return (mu, var)

    def get_correlation_metric(self, delta):
        raise Exception("TODO: get_correlation_metric not implemented")

    def gen_model_from_hyperparameters(self, data : Data, hyperparameters : dict, modeling_options : dict, **kwargs):
        import george
        if kwargs['model_random_seed'] != None:
            seed = kwargs['model_random_seed']
            if data.P is not None:
                for P_ in data.P:
                    seed += len(P_)
            np.random.seed(seed)

        if modeling_options["multitask"] == "yes":
            raise Exception("TODO: IMPLEMENT multitask")
        else:
            input_dim = data.P.shape[1] if data.P.ndim > 1 else 1
            intialguess = hyperparameters["noise_variance"] + hyperparameters["variance"] + hyperparameters["lengthscale"]
            if modeling_options['model_kern'] == 'RBF':
                K = george.kernels.ExpSquaredKernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude 
            elif modeling_options['model_kern'] == 'Matern32':
                K = george.kernels.Matern32Kernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude
            elif modeling_options['model_kern'] == 'Matern52':
                K = george.kernels.Matern52Kernel(metric=np.array(intialguess[2:]), ndim=input_dim)
                amplitude = intialguess[1]
                K *= amplitude  
            else:
                raise Exception("TODO: IMPLEMENT OTHER KERNELS")

            if modeling_options['model_lowrank'] == 'yes':
                kwargs_variable = {
                    'min_size': kwargs['model_hodlrleaf'],
                    'tol': kwargs['model_hodlrtol'],
                    'seed': 42
                }
                self.M = george.GP(kernel=K, white_noise=np.log(intialguess[0]), fit_white_noise=True, solver=george.solvers.HODLRSolver,**kwargs_variable)
            else:
                self.M = george.GP(kernel=K, white_noise=np.log(intialguess[0]), fit_white_noise=True, solver=george.solvers.BasicSolver)                
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

