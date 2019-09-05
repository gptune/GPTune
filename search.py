class Search(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer):

        self.problem = problem
        self.computer = computer

    @abstractmethod
    def search(self, data : Data, model : Model, tid : int, **kwargs) -> np.ndarray:

        raise Exception("Abstract method")

    @abstractmethod
    def search_multitask(self, data : Data, model : Model, tids = None : Collection[int], **kwargs) -> Collection[np.ndarray]:

        raise Exception("Abstract method")

class SearchPyGMO(Search):

    """
    XXX: This class, together with the underlying PyGMO only works on Intel-based CPUs.
    The reason is that PyGMO requires the Intel 'Thread Building Block' library to compile and execute.
    """

    class SurrogateProblem(object):

        def __init__(self, problem, computer, data, model, tid):

            self.problem = problem
            self.computer = computer
            self.data = data
            self.model = model

            self.tid = tid

            self.t     = self.data.T[tid]
            self.XOrig = self.data.X[tid]

        def get_bounds(self):

            DP = self.problem.DP

            return ([0. for i in range(DP)], [1. for  i in range(DP)])

        # Acquisition function
        def ei(self, x):

            """ Expected Improvement """

            ymin = self.data.Y[self.tid].min()
            (mu, var) = self.model.predict(x, tid=self.tid)
            mu = mu[0][0]
            var = max(1e-18, var[0][0])
            std = np.sqrt(var)
            chi = (ymin - mu) / std
            Phi = 0.5 * (1.0 + sp.special.erf(chi / np.sqrt(2)))
            phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * var)
            EI = (ymin - mu) * Phi + var * phi

            return EI

        def fitness(self, x):

            xi = self.problem.PS.inverse_transform(x)
            if (any(np.array_equal(xx, xi) for xx in self.XOrig)):
                cond = False
            else:
                kwargs = tuner.point2kwargs(self.t, xi) # XXX
                cond = tuner.check_conditions(kwargs)  # XXX
            if (cond):
                return (- self.ei(x),)
            else:
                return (float("Inf"),)

    def search(self, data : Data, model : Model, tid : int, **kwargs) -> np.ndarray:

#        search_n_threads
#        search_n_processes = 1

#        search_algo
#        search_udi
#        search_population_size = 100
#        search_n_generations = 100
#        search_n_evolve = 10
#        search_n_max_iters = 100

        #search_algo = 'pso'
        #search_algo = 'cmaes'

        #search_udi = 'thread_island'
        #search_udi = 'mp_island' #XXX advise the user not to use this kind of udi as it lunches several processes and deadlock on some weird semaphores
        #search_udi = 'ipyparallel_island' #XXX advise the user not to use this kind of udi as it is not tested

        prob = SurrogateProblem(self.problem, self.computer, data, model, tid)

        try:
            algo = eval(f'pg.{search_algo}(gen = search_n_generations)')
        except:
            raise Exception(f'Unknown optimization algorithm "{search_algo}"')

        try:
            udi = eval(f'pg.{search_udi}()')
        except:
            raise Exception('Unknown user-defined-island "{search_udi}"')

        bestX = []
        cond = False
        cpt = 0
        while (not cond and cpt < search_n_max_iters):
            archi = pg.archipelago(n = search_n_threads, prob = prob, algo = algo, udi = udi, pop_size = search_population_size)
            archi.evolve(n = search_n_evolve)
            archi.wait()
            champions_f = archi.get_champions_f()
            champions_x = archi.get_champions_x()
            indexes = list(range(len(champions_f)))
            indexes.sort(key=champions_f.__getitem__)
            for idx in indexes:
                if (champions_f[idx] < float('Inf')):
                    cond = True
                    bestX.append(self.problem.PS.inverse_transform(champions_x[idx]).reshape(1, self.problem.DP))
                    break
            cpt += 1

        if (verbose):
            print(tid, 'OK' if cond else 'KO'); sys.stdout.flush()

        return (tid, bestX)

    def search_multitask(self, data : Data, model : Model, tids = None : Collection[int], **kwargs) -> Collection[np.ndarray]:

        if (parallelism_method not in ["Sequential", "Thread", "Process", "MPI"]):

            raise Exception("Unknown parallelism method")

        t1 = time.time()

        if (parallelism_method == "MPI"):

            if (tids is None):
                tids = list(range(self.mpi_rank, self.NT, self.mpi_size))
                print(self.mpi_rank, tids)

            res = self.search(tids=tids, parallelism_method="Process")

            res = mpi_comm.gather(res, root=0)
            if (self.mpi_rank == 0):
                res = list(itertools.chain(*res))
                res.sort(key = lambda x : x[0])
            else:
                res = None
            res = mpi_comm.bcast(res, root=0)

        else:

            if (tids is None):
                tids = list(range(self.NT))

            if (parallelism_method == "Sequential"):

                res = list(map(self.search_optima, tids))

            else:

                if (parallelism_method == "Thread"):

                    nth = 32 #XXX set to number of threads per (MPI) process or number of available cores per node
                    kwargs = {"max_workers":nth}
                    Executor = concurrent.futures.ThreadPoolExecutor

                elif (parallelism_method == "Process"):

                    nproc = 1 #XXX set to number of threads per (MPI) process or number of available cores per node
                    kwargs = {"max_workers":nproc}
                    Executor = concurrent.futures.ProcessPoolExecutor

                with Executor(**kwargs) as executor:
                    res = list(executor.map(self.search_optima, tids, timeout=None, chunksize=1))

            res.sort(key = lambda x : x[0])


        t2 = time.time()
        if (verbose and self.mpi_rank == 0):
            print("Search time: %f s"%(t2 - t1))

        return res

