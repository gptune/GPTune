class Options(Mapping):

    def __init__(self, **kwargs):

        # GPTune

        distributed_memory_parallelism = False
        shared_memory_parallelism      = False

        constraints_evaluation_parallelism = False
        objective_evaluation_parallelism   = False

        # Sample

        sample = 'SampleOpenTURNS' # Default sample class

        sample_max_iters = 10**9

        # Model

        model = 'Model_LCM' # Default model class

        model_threads = 1
        model_processes = 1
        model_groups = 1

        model_restarts = 1
        model_max_iters = 15000
        model_latent = 0
        model_sparse = False
        model_inducing = None
        model_layers = 2

        # Search

        search = 'SearchPyGMO' # Default search class

        search_threads = 1
        search_processes = 1

        search_algo = 'pso' # ['pso', 'cmaes']
        search_udi = 'thread_island' # ['thread_island', 'mp_island', 'ipyparallel_island']
        #XXX 'mp_island' : advise the user not to use this kind of udi as it launches several processes and deadlock on some weird semaphores
        #XXX 'ipyparallel_island' : advise the user not to use this kind of udi as it is not tested
        search_pop_size = 100
        search_gen = 100
        search_evolve = 10
        search_max_iters = 100

        self.update(kwargs)

