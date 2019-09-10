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

    def search_multitask(self, data : Data, model : Model, tids = None : Collection[int], i_am_manager = True : bool, **kwargs) -> Collection[np.ndarray]:

#        if (kwargs['search_processes'] == 1):
#        if (kwargs['search_threads'] == 1):
#        if (parallelism_method == "MPI"):
#        if (parallelism_method == "Sequential"):
#        if (parallelism_method == "Thread"):
#        if (parallelism_method == "Process"):

        if (tids is None):

            if (kwargs['distributed_memory_parallelism']):
                tids = list(range(self.mpi_rank, self.NT, self.mpi_size))
            else:
                tids = list(range(self.NT))

        if (not kwargs['shared_memory_parallelism']):

            res = list(map(self.search_optima, tids))

        else:

            kwargs = {"max_workers":kwargs['search_threads']}
            Executor = concurrent.futures.ThreadPoolExecutor
            #Executor = concurrent.futures.ProcessPoolExecutor

            with Executor(**kwargs) as executor:
                fun = function.partial(self.search, data = data, model = model, kwargs = kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize=1))

            res.sort(key = lambda x : x[0])

        if (kwargs['distributed_memory_parallelism']):

            res = mpi_comm.gather(res, root=0)
            if (self.mpi_rank == 0):
                res = list(itertools.chain(*res))
                res.sort(key = lambda x : x[0])
            else:
                res = None
            res = mpi_comm.bcast(res, root=0)

        return res

if __name__ == '__main__':

    comm = MPI.Comm.Get_parent().Merge()
    (searcher, data, model, tids, kwargs) = comm.bcast(None, root=0)
    searcher.search_multitask(data, model, tids, i_am_manager = False, **kwargs)
    comm.Disconnect()

