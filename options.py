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

from typing import Mapping

class Options(Mapping):

    def __init__(self, **kwargs):

        # GPTune

        distributed_memory_parallelism = False
        shared_memory_parallelism      = False

        constraints_evaluation_parallelism = False
        objective_evaluation_parallelism   = False

        verbose = False

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
        search_multitask_threads = 1
        search_multitask_processes = 1

        search_algo = 'pso' # ['pso', 'cmaes']
        search_udi = 'thread_island' # ['thread_island', 'mp_island', 'ipyparallel_island']
        #XXX 'mp_island' : advise the user not to use this kind of udi as it launches several processes and deadlock on some weird semaphores
        #XXX 'ipyparallel_island' : advise the user not to use this kind of udi as it is not tested
        search_pop_size = 100
        search_gen = 100
        search_evolve = 10
        search_max_iters = 100

        self.update(kwargs)

