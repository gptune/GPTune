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

class Options(dict):

    def __init__(self, **kwargs):

        # GPTune

        distributed_memory_parallelism = False
        shared_memory_parallelism      = False

        constraints_evaluation_parallelism = False
        objective_evaluation_parallelism   = False

        verbose = True

        # Sample

        sample_class = 'SampleLHSMDU' # Default sample class
        #sample_class = 'SampleOpenTURNS'

        #sample_algo = None
        sample_algo = 'LHS-MDU' #Latin hypercube sampling with multidimensional uniformity
        #sample_algo = 'MCS'     #Monte Carlo Sampling

        sample_max_iter = 10**9

        # Model

        #model_class = 'Model_GPy_LCM' # Default model class
        model_class = 'Model_LCM'

        model_threads = 1
        model_processes = 1
        model_groups = 1

        model_restarts = 1
        model_max_iters = 15000
        model_latent = None
        model_sparse = False
        model_inducing = None
        model_layers = 2

        # Search

        search_class = 'SearchPyGMO' # Default search class

        search_threads = 1
        search_processes = 1
        search_multitask_threads = 1
        search_multitask_processes = 1

        search_algo = 'pso' # ['pso', 'cmaes']
        search_udi = 'thread_island' # ['thread_island', 'mp_island', 'ipyparallel_island']
        #XXX 'mp_island' : advise the user not to use this kind of udi as it launches several processes and deadlock on some weird semaphores
        #XXX 'ipyparallel_island' : advise the user not to use this kind of udi as it is not tested
        search_pop_size = 1000
        search_gen = 1000
        search_evolve = 10
        search_max_iters = 10

        self.update(locals())
        self.update(kwargs)
        self.pop('self')

