class Model(abc.ABC):

    @abstractmethod
    def __init__(self, problem : Problem, **kwargs):

        self.problem = problem
        self.M = None

    @abstractmethod
    def train(self, data : Data, **kwargs):

        raise Exception("Abstract method")

    @abstractmethod
    def update(self, newdata : Data, do_train = False: bool, **kwargs):

        raise Exception("Abstract method")

    @abstractmethod
    def predict(self, points : Collection[np.ndarrays], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

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

        multitask = len(data.T) > 1

        if (model_sparse and model_inducing is None):
            if (multitask):
                lenx = sum([len(X) for X in data.X])
            else:
                lenx = len(data.X)
            model_inducing = int(min(lenx, 3 * np.sqrt(lenx)))

        if (multitask):
            kernels_list = [GPy.kern.RBF(input_dim = self.problem.DP, ARD=True) for k in range(model_latent)]
            K = GPy.util.multioutput.LCM(input_dim = self.problem.DP, num_outputs = data.NI, kernels_list = kernels_list, W_rank = 1, name='GPy_LCM')
            if (model_sparse):
                self.M = GPy.models.SparseGPCoregionalizedRegression(X_list = data.X, Y_list = data.Y, kernel = K, num_inducing = model_inducing)
            else:
                self.M = GPy.models.GPCoregionalizedRegression(X_list = data.X, Y_list = data.Y, kernel = K)
        else:
            K = GPy.kern.RBF(input_dim = self.problem.DP, ARD=True, name='GPy_GP')
            if (model_sparse):
                self.M = GPy.models.SparseGPRegression(data.X[0], data.Y[0], kernel = K, num_inducing = model_inducing)
            else:
                self.M = GPy.models.GPRegression(data.X[0], data.Y[0], kernel = K)
            
        np.random.seed(mpi_rank)
        num_restarts = max(1, model_n_restarts // mpi_size)

        resopt = self.M.optimize_restarts(num_restarts = num_restarts, robust = True, verbose = verbose, parallel = (model_n_threads > 1), num_processes = model_n_threads, messages = "True", optimizer = 'lbfgs', start = None, max_iters = model_max_iters, ipython_notebook = False, clear_after_finish = True)

        self.M.param_array[:] = allreduce_best(self.M.param_array[:], resopt)[:]
        self.M.parameters_changed()

        return

    def update(self, newdata : Data, do_train = False: bool, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarrays], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        (mu, var) = self.M.predict_noiseless(x)

        return (mu, var)


class Model_LCM(Model):

    def train(self, data : Data, **kwargs):

        
        self.M = None
        if (num_restarts is None):
            num_restarts = num_subgroups

        num_subgroups = mpi_size / model_num_processes
        color = self.mpi_rank // (self.mpi_size // num_subgroups)
        key   = self.mpi_rank %  (self.mpi_size // num_subgroups)
        subcomm = mpi_comm.Split(color, key)

        np.random.seed(color)
        if ((self.M is not None) and (color == 0)):
            ker = self.M.kern
            (bestxopt, bestfopt) = ker.train_kernel(self.X, self.Y, mpi_comm=subcomm, verbose=self.verbose)
            color += num_subgroups
        else:
            (bestxopt, bestfopt) = (None, float('Inf'))
        for i in range(color, num_restarts, num_subgroups):
            ker = LMC(input_dim=self.DI, num_outputs=self.NT, Q=Q)
            (xopt, fopt) = ker.train_kernel(data.X, data.Y, mpi_comm=subcomm, verbose=verbose)
            if (fopt < bestfopt):
                bestxopt = xopt
                bestfopt = fopt

        mpi_comm.Barrier()
        K = LMC(input_dim=self.DI, num_outputs=self.NT, Q=Q)
        bestxopt = allreduce_best(bestxopt, bestfopt)
        K.set_param_array(bestxopt)

        likelihoods_list = [GPy.likelihoods.Gaussian(variance = K.sigma[i], name="Gaussian_noise_%s" %i) for i in range(data.NT)]
        self.M = GPy.models.GPCoregionalizedRegression(data.X, data.Y, K, likelihoods_list=likelihoods_list)

        return

    def update(self, newdata : Data, do_train = False: bool, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarrays], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        x = np.empty((1, points.shape[0] + 1))
        x[0,:-1] = points
        x[0,-1] = tid
        (mu, var) = self.M.predict_noiseless(x)

        return (mu, var)


class Model_DGP(Model):

    def train(self, data : Data, **kwargs):

        multitask = len(self.T) > 1

        if (multitask):
            X = np.array([np.concatenate((self.T[i], self.X[i][j])) for i in range(len(self.T)) for j in range(self.X[i].shape[0])])
        else:
            X = self.X[0]
        Y = np.array(list(itertools.chain.from_iterable(self.Y)))

        #--------- Model Construction ----------#
        model_n_layers = 2
        # Define what kernels to use per layer
        kerns = [GPy.kern.RBF(input_dim=Q, ARD=True) + GPy.kern.Bias(input_dim=Q) for lev in range(model_n_layers)]
        kerns.append(GPy.kern.RBF(input_dim=X.shape[1], ARD=True) + GPy.kern.Bias(input_dim=X.shape[1]))
        # Number of inducing points to use
        if (num_inducing is None):
            if (multitask):
                lenx = sum([len(X) for X in self.X])
            else:
                lenx = len(self.X)
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

    def update(self, newdata : Data, do_train = False: bool, **kwargs):

        #XXX TODO
        self.train(newdata, **kwargs)

    def predict(self, points : Collection[np.ndarrays], tid : int, **kwargs) -> Collection[Tuple[float, float]]:

        (mu, var) = self.M.predict(np.concatenate((self.T[tid], x)).reshape((1, self.DT + self.DI)))

        return (mu, var)

