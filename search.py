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

class Search(abc.ABC):

    def __init__(self, problem : Problem, computer : Computer):

        self.problem = problem
        self.computer = computer

    @abstractmethod
    def search(self, data : Data, model : Model, tid : int, **kwargs) -> np.ndarray:

        raise Exception("Abstract method")

    @abstractmethod
    def search_multitask(self, data : Data, model : Model, tids = None : Collection[int], i_am_manager = True : bool, **kwargs) -> Collection[np.ndarray]:

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

#        if (kwargs['distributed_memory_parallelism']):
#
#            if (i_am_manager):
#
#                mpi_comm = self.computer.spawn(__file__, kwargs['search_processes'], kwargs['search_threads'], args=self, kwargs=) # XXX add args and kwargs
#                res = mpi_comm.gather(None, root=0)
#
#            else:
#
#                mpi_comm = kwargs['mpi_comm']
#                mpi_rank = mpi_comm.Get_rank()
#                mpi_size = mpi_comm.Get_size()
#
#        if (tids is None):
#
#            if (kwargs['distributed_memory_parallelism']):
#                tids = list(range(self.mpi_rank, self.NT, self.mpi_size))
#            else:
#                tids = list(range(self.NT))

        if (kwargs['distributed_memory_parallelism'] and i_am_manager):

            with mpi4py.futures.MPIPoolExecutor(max_workers = kwargs['search_multitask_processes']) as executor:
                fun = function.partial(self.search_multitask, data = data, model = model, i_am_manager = False, kwargs = kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize = kwargs['search_multitask_threads']))

        elif (kwargs['shared_memory_parallelism']):
            
            #with concurrent.futures.ProcessPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers = kwargs['search_multitask_threads']) as executor:
                fun = function.partial(self.search, data = data, model = model, kwargs = kwargs)
                res = list(executor.map(fun, tids, timeout=None, chunksize=1))

        else:

            fun = function.partial(self.search, data = data, model = model, kwargs = kwargs)
            res = list(map(self.search, tids))

        res.sort(key = lambda x : x[0])

        return res

#if __name__ == '__main__':
#
#    comm = MPI.Comm.Get_parent().Merge()
#    (searcher, data, model, tids, kwargs) = comm.bcast(None, root=0)
#    searcher.search_multitask(data, model, tids, i_am_manager = False, **kwargs)
#    comm.Disconnect()

