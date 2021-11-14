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

import math

class Options(dict):

    def __init__(self, **kwargs):

        """ Options for GPTune """
        RCI_mode = False         # whether the reverse communication mode will be used
        mpi_comm = None          # The mpi communicator that invokes gptune if mpi4py is installed
        distributed_memory_parallelism = False   # Using distributed_memory_parallelism for the modeling (one MPI per model restart) and search phase (one MPI per task)
        shared_memory_parallelism      = False   # Using shared_memory_parallelism for the modeling (one MPI per model restart) and search phase (one MPI per task)
        constraints_evaluation_parallelism = False  # Reserved option
        verbose = False     # Control the verbosity level
        oversubscribe = False     # Set this to True when the physical core count is less than computer.nodes*computer.cores and the --oversubscribe MPI runtime option is used


        """ Options for the function evaluation """
        objective_evaluation_parallelism   = False  # Using distributed_memory_parallelism or shared_memory_parallelism for evaluating multiple application instances in parallel
        objective_multisample_processes = None  # Number of MPIs each handling one application call
        objective_multisample_threads = None  # Number of threads each handling one application call
        objective_nprocmax = None # Maximum number of cores for each application call, default to computer.cores*computer.nodes-1

        """ Options for the sampling phase """
        sample_class = 'SampleLHSMDU' # Supported sample classes: 'SampleLHSMDU', 'SampleOpenTURNS'
        sample_algo = 'LHS-MDU' # Supported sample algorithms in 'SampleLHSMDU': 'LHS-MDU' --Latin hypercube sampling with multidimensional uniformity, 'MCS' --Monte Carlo Sampling
        sample_max_iter = 10**9  # Maximum number of iterations for generating random samples and testing the constraints



        """ Options for the modeling phase """
        model_class = 'Model_LCM' # Supported sample algorithms: 'Model_GPy_LCM' -- LCM from GPy, 'Model_LCM' -- LCM with fast and parallel inversion, 'Model_DGP' -- deep Gaussian process
        model_output_constraint = False # True: if Model_LCM is used, check output range constraint and disregard out-of-range output (put a large value)
        model_threads = None  # Number of threads used for building one GP model in Model_LCM
        model_processes = None # Number of MPIs used for building one GP model in Model_LCM
        model_groups = 1  # Reserved option
        model_restarts = 1 # Number of random starts each building one initial GP model
        model_restart_processes = None  # Number of MPIs each handling one random start
        model_restart_threads = None   # Number of threads each handling one random start
        model_max_iters = 15000   # Number of maximum iterations for the optimizers
        model_jitter = 1e-10   # Initial jittering
        model_latent = None # Number of latent functions for building one LCM model, defaults to number of tasks
        model_sparse = False # Whether to use SparseGPRegression or SparseGPCoregionalizedRegression from Model_GPy_LCM
        model_inducing = None # Number of inducing points for SparseGPRegression or SparseGPCoregionalizedRegression
        model_layers = 2 # Number of layers for Model_DGP
        model_max_jitter_try = 10 # Max number of jittering 


        """ Options for the search phase """
        search_class = 'SearchPyGMO' #'SearchCMO' #'SearchPyGMO' # Supported searcher classes: 'SearchPyGMO'
        search_threads = None  # Number of threads in each thread group handling one task
        search_processes = 1  # Reserved option
        search_multitask_threads = None # Number of threads groups each handling one task
        search_multitask_processes = None # Number of MPIs each handling one task
        search_algo = 'pso' # Supported search algorithm in pygmo: single-objective: 'pso' -- particle swarm, 'cmaes' -- covariance matrix adaptation evolution. multi-objective 'nsga2' -- Non-dominated Sorting GA, 'nspso' -- Non-dominated Sorting PSO, 'maco' -- Multi-objective Hypervolume-based ACO, 'moead' -- Multi-objective EA vith Decomposition
        search_udi = 'thread_island' # Supported UDI options for pgymo: 'thread_island' --Thread island, 'ipyparallel_island' --Ipyparallel island
        search_pop_size = 1000 # Population size in pgymo
        search_gen = 1000  # Number of evolution generations in pgymo
        search_evolve = 10  # Number of times migration in pgymo
        search_max_iters = 10  # Max number of searches to get results respecting the constraints
        search_more_samples = 1  # Maximum number of points selected using a multi-objective search algorithm


        """ Options for the multi-arm bandit algorithm """
        budget_min = 0.1 # minimum budget
        budget_max = 1 # maximum budget
        budget_base = 2 # the number of arms is floor{log_{budget_base}{budget_max/budget_min}}+1
        fidelity_map = None

        self.update(locals())
        self.update(kwargs)
        self.pop('self')

        self.update(locals())
        self.update(kwargs)
        self.pop('self')


    def validate(self, computer, **kwargs):

        """  modify the options as needed """
        if ((self['model_class']=='Model_LCM' or self['model_class']=='Model_LCM_constrained') and self['RCI_mode']==True):
            self['model_class']='Model_GPy_LCM'

        if (self['distributed_memory_parallelism'] and self['shared_memory_parallelism']):
            self['shared_memory_parallelism']=False

        if (self['distributed_memory_parallelism']):
            if(self['search_multitask_processes'] is None):
                self['search_multitask_processes'] = max(1,computer.cores*computer.nodes-1) # computer.nodes
            self['search_multitask_threads'] = 1
            if(self['model_restart_processes'] is None):
                self['model_restart_processes'] = self['model_restarts']
            self['model_restart_processes'] = min(self['model_restart_processes'],max(1,computer.cores*computer.nodes-2))
            self['model_restart_threads'] = 1
        elif(self['shared_memory_parallelism']):
            self['search_multitask_processes'] = 1
            if(self['search_multitask_threads'] is None):
                self['search_multitask_threads'] = computer.cores
            self['model_restart_processes'] = 1
            if(self['model_restart_threads'] is None):
                self['model_restart_threads'] = self['model_restarts']
            self['model_restart_threads'] = min(self['model_restart_threads'],computer.cores)
        else:
            self['search_multitask_processes'] = 1
            self['search_multitask_threads'] = 1
            self['model_restart_processes'] = 1
            self['model_restart_threads'] = 1

        if (self['model_class']=='Model_LCM' or self['model_class']=='Model_LCM_constrained'):
            if(self['model_processes'] is None):
                if (self['distributed_memory_parallelism']):
                    self['model_processes'] = max(1,math.floor(((computer.cores*computer.nodes-1)/(self['model_restart_processes'])-1)/self['model_restart_threads']))
                else:
                    self['model_processes'] = max(1,math.floor((computer.cores*computer.nodes-1)/self['model_restart_processes']/self['model_restart_threads']))
            self['model_threads'] =1
        else:
            self['model_processes'] = 1
            self['model_threads'] = 1  # YL: this requires more thoughts, maybe math.floor((computer.cores)/self['model_restart_threads'])

        if(self['search_threads'] is None):
            # if (self['distributed_memory_parallelism']):
            #   self['search_threads'] = min(computer.cores,max(1,math.floor((computer.cores*computer.nodes-1)/self['search_multitask_processes'])))
            # else:
            #   self['search_threads'] = min(computer.cores,max(1,math.floor((computer.cores*computer.nodes)/self['search_multitask_threads'])))
            self['search_threads'] =1

        if(self['objective_nprocmax'] is None):
            self['objective_nprocmax'] = computer.cores*computer.nodes-1
        self['objective_nprocmax'] = min(self['objective_nprocmax'],computer.cores*computer.nodes-1)


        if (self['objective_evaluation_parallelism']==True and self['distributed_memory_parallelism']==True):
            self['objective_nprocmax'] = max(1,min(self['objective_nprocmax'],computer.cores*computer.nodes-2))
            nproc = max(1,math.floor((computer.cores*computer.nodes-1)/(self['objective_nprocmax']+1)))  # here we always assume the user invoke application code with MPI_Spawn, if not, "+1" can be removed
            if(self['objective_multisample_processes'] is None):
                self['objective_multisample_processes'] = nproc
            self['objective_multisample_processes'] = min(self['objective_multisample_processes'],nproc)
            self['objective_multisample_threads'] = 1
        elif (self['objective_evaluation_parallelism']==True and self['shared_memory_parallelism']==True):
            nproc = max(1,math.floor((computer.cores*computer.nodes)/(self['objective_nprocmax']+1)))
            if(self['objective_multisample_threads'] is None):
                self['objective_multisample_threads'] = computer.cores
            self['objective_multisample_threads'] = min(self['objective_multisample_threads'],computer.cores,nproc)
            self['objective_multisample_processes'] = 1
        else:
            self['objective_multisample_threads'] = 1
            self['objective_multisample_processes'] = 1

        if(self['verbose']==True):
            print('\n\n------Validating the options')
            print("  ")
            print("  total core counts provided to GPTune:", computer.cores*computer.nodes)
            print("   ---> distributed_memory_parallelism:", self['distributed_memory_parallelism'])
            print("   ---> shared_memory_parallelism:", self['shared_memory_parallelism'])
            print("   ---> objective_evaluation_parallelism:", self['objective_evaluation_parallelism'])


        if(self['distributed_memory_parallelism']):
            ncore_model = (self['model_processes']+1)*self['model_threads']*self['model_restart_processes']+1
        else:
            ncore_model = (self['model_processes']+1)*self['model_threads']*self['model_restart_threads']

        if(self['verbose']==True):    
            print("  ")
            print("  total core counts for modeling:", ncore_model)
            print("   ---> model_processes:", self['model_processes'])
            print("   ---> model_threads:", self['model_threads'])
            print("   ---> model_restart_processes:", self['model_restart_processes'])
            print("   ---> model_restart_threads:", self['model_restart_threads'])


        if(self['distributed_memory_parallelism']):
            ncore_search = self['search_threads']*self['search_multitask_processes']+1
        else:
            ncore_search = self['search_threads']*(self['search_multitask_threads'])

        if(self['verbose']==True):        
            print("  ")
            print("  total core counts for search:", ncore_search)
            print("   ---> search_processes:", self['search_processes'])
            print("   ---> search_threads:", self['search_threads'])
            print("   ---> search_multitask_processes:", self['search_multitask_processes'])
            print("   ---> search_multitask_threads:", self['search_multitask_threads'])

        if(self['distributed_memory_parallelism']):
            if(self['objective_evaluation_parallelism']==True):
                ncore_obj = self['objective_multisample_processes']*(self['objective_nprocmax']+1)+1
            else:
                ncore_obj = (self['objective_nprocmax']+1)
        else:
            ncore_obj = self['objective_multisample_threads']*(self['objective_nprocmax']+1)
        if(self['verbose']==True):        
            print("  ")
            print("  total core counts for objective function evaluation:", ncore_obj)
            print("   ---> core counts in a single application run:", self['objective_nprocmax'])
            print("   ---> objective_multisample_processes:", self['objective_multisample_processes'])
            print("   ---> objective_multisample_threads:", self['objective_multisample_threads'])

        if(self['oversubscribe']==False):
            if (computer.cores*computer.nodes<=1):
                raise Exception("the computer should has at least 2 total cores")

            if ((computer.cores*computer.nodes)<ncore_model):
                raise Exception("Reduce one of the options: model_restart_processes,model_restart_threads,model_processes,model_threads")
            if ((computer.cores*computer.nodes)<ncore_search):
                raise Exception("Reduce one of the options: search_multitask_processes,search_multitask_threads,search_processes,search_threads")
            if ((computer.cores*computer.nodes)<ncore_obj):
                raise Exception("Reduce one of the options: objective_multisample_processes,objective_multisample_threads,objective_nprocmax")

