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
import os

class Options(dict):

    def __init__(self, **kwargs):

        """ Options for GPTune """
        lite_mode = False        # whether to disable all C/C++ dependencies 
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
        objective_nospawn = False # Whether the application code is launched via MPI spawn. If True, self['objective_nprocmax'] cores are used per function evaluation, otherwise self['objective_nprocmax']+1 cores are used. 

        """ Options for the sampling phase """
        sample_class = 'SampleLHSMDU' # Supported sample classes: 'SampleLHSMDU', 'SampleOpenTURNS'
        sample_algo = 'LHS-MDU' # Supported sample algorithms in 'SampleLHSMDU': 'LHS-MDU' --Latin hypercube sampling with multidimensional uniformity, 'MCS' --Monte Carlo Sampling
        sample_max_iter = 10**9  # Maximum number of iterations for generating random samples and testing the constraints
        sample_random_seed = None # Specify a certain random seed for the pilot sampling phase


        """ Options for the modeling phase """
        model_class = 'Model_LCM' # Supported sample algorithms: 'Model_GPy_LCM' -- LCM from GPy, 'Model_LCM' -- LCM with fast and parallel inversion, 'Model_DGP' -- deep Gaussian process
        model_kern = 'RBF' # Supported kernels in 'Model_GPy_LCM' model class option -- 'RBF', 'Exponential' or 'Matern12', 'Matern32', 'Matern52'
        model_output_constraint = None # Check output range constraints and disregard out-of-range outputs. Supported options: 'LargeNum': Put a large number, 'Ignore': Ignore those configurations, None: do not check out-of-range outputs.
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
        model_random_seed = None # Specify a certain random seed for the surrogate modeling phase


        """ Options for the search phase """
        search_class = 'SearchPyGMO' # Supported searcher classes: 'SearchPyGMO', 'SearchCMO', 'SearchSciPy', 'SearchPyMoo' 
        search_threads = None  # Number of threads in each thread group handling one task
        search_processes = 1  # Reserved option
        search_multitask_threads = None # Number of threads groups each handling one task
        search_multitask_processes = None # Number of MPIs each handling one task
        search_algo = 'pso' # Supported search algorithms:
            # 'SearchPyGMO' or 'SearchCMO': single-objective: 'pso' -- particle swarm, 'cmaes' -- covariance matrix adaptation evolution. multi-objective 'nsga2' -- Non-dominated Sorting GA, 'nspso' -- Non-dominated Sorting PSO, 'maco' -- Multi-objective Hypervolume-based ACO, 'moead' -- Multi-objective EA vith Decomposition. 

            # 'SearchSciPy': single-objective: 'l-bfgs-b', 'dual_annealing', 'trust-constr', 'shgo'

            # 'SearchPyMoo': single-objective: 'pso' -- particle swarm, 'ga' -- genetic algorithm. multi-objective 'nsga2' -- Non-dominated Sorting GA, 'moead' -- Multi-objective EA vith Decomposition. 

        search_udi = 'thread_island' # Supported UDI options for pgymo: 'thread_island' --Thread island, 'ipyparallel_island' --Ipyparallel island
        search_pop_size = 1000 # Population size in pgymo or pymoo
        search_gen = 100  # Number of evolution generations in pgymo or pymoo
        search_evolve = 10  # Number of times migration in pgymo 
        search_max_iters = 10  # Max number of searches to get results respecting the constraints
        search_more_samples = 1  # Maximum number of points selected using a multi-objective search algorithm
        search_random_seed = None # Specify a certain random seed for the search phase (it works for only SearchPyGMO option for now)

        """ Options for transfer learning """
        TLA_method = 'Regression' #"LCM_BF" #'Sum' #'regression_weights_no_scale'
        regression_log_name = 'models_weights.log'

        """ Options for the multi-arm bandit algorithm """
        budget_min = 0.1 # minimum budget
        budget_max = 1 # maximum budget
        budget_base = 2 # the number of arms is floor{log_{budget_base}{budget_max/budget_min}}+1
        fidelity_map = None

        """ Options for cGP """
        N_PILOT_CGP=20      # number of initial samples
        N_SEQUENTIAL_CGP=20 # number of sequential samples
        RND_SEED_CGP=1      # random seed (int)
        EXAMPLE_NAME_CGP='obj_name_dummy' # name of the objective function for logging purpose     
        METHOD_CGP='FREQUENTIST' # 'FREQUENTIST' or 'BAYESIAN
        #parameters HMC Bayesian sampling when METHOD_CGP='BAYESIAN'
        N_BURNIN_CGP=500
        N_MCMCSAMPLES_CGP=500
        N_INFERENCE_CGP=500
        EXPLORATION_RATE_CGP=1.0 #Exploration rate is the probability (between 0 and 1) of following the next step produced by acquisition function.
        NO_CLUSTER_CGP=False #If NO_CLUSTER = True, a simple GP will be used.
        N_NEIGHBORS_CGP=3 # number of neighbors for deciding cluster components  
        CLUSTER_METHOD_CGP='BGM' #Cluster method: BGM or KMeans
        N_COMPONENTS_CGP=3 # maximal number of clusters
        ACQUISITION_CGP='EI' #acquisition function: EI or MSPE
        BIGVAL_CGP=1e12 #return this big value in the objective function when constraints are not respected

        self.update(locals())
        self.update(kwargs)
        self.pop('self')

        self.update(locals())
        self.update(kwargs)
        self.pop('self')


    def validate(self, computer, **kwargs):

        """  modify the options as needed """

        if (os.environ.get('GPTUNE_LITE_MODE') is not None):
            self['lite_mode']=True

        if(self['lite_mode']==True):
            self['model_class']='Model_GPy_LCM'
            self['distributed_memory_parallelism']=False
            self['shared_memory_parallelism']=False
            self['objective_evaluation_parallelism']=False
            self['objective_nospawn']=True
            self['sample_class']='SampleLHSMDU'
            self['sample_algo']='LHS-MDU'
            self['model_threads']=1
            self['model_processes']=1
            self['model_restart_processes']=1
            self['model_restart_threads']=1
            self['search_multitask_threads']=1
            self['search_multitask_processes']=1
            self['search_threads']=1
            self['search_more_samples']=1
            
            # use 'SearchSciPy' to replace 'SearchPyGMO' if single-objective
            # use 'SearchPyMoo' to replace 'SearchPyGMO' if multi-objective
            if((self['search_class']=='SearchPyGMO' or self['search_class']=='SearchCMO') and (self["search_algo"] == 'pso' or self["search_algo"] == 'cmaes')):
                self['search_class']='SearchSciPy'
            if((self['search_class']=='SearchPyGMO' or self['search_class']=='SearchCMO') and (self["search_algo"] == 'nsga2' or self["search_algo"] == 'nspso' or self["search_algo"] == 'maco' or self["search_algo"] == 'moead')):
                self['search_class']='SearchPyMoo'               

            # set the default search algorithm in 'SearchSciPy' 
            if(self['search_class']=='SearchSciPy' and not (self["search_algo"] == 'trust-constr' or self["search_algo"] == 'l-bfgs-b' or self["search_algo"] == 'dual_annealing')):
                self["search_algo"]='trust-constr'

            # set the default search algorithm in 'SearchPyMoo' 
            if(self['search_class']=='SearchPyMoo' and not (self["search_algo"] == 'nsga2' or self["search_algo"] == 'moead')):
                self["search_algo"]='nsga2'

        else:
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
                if(self['objective_nospawn']==True):
                    self['objective_nprocmax'] = max(1,min(self['objective_nprocmax'],computer.cores*computer.nodes-1))
                    nproc = max(1,math.floor((computer.cores*computer.nodes-1)/(self['objective_nprocmax'])))
                else:
                    self['objective_nprocmax'] = max(1,min(self['objective_nprocmax'],computer.cores*computer.nodes-2))
                    nproc = max(1,math.floor((computer.cores*computer.nodes-1)/(self['objective_nprocmax']+1)))                
                if(self['objective_multisample_processes'] is None):
                    self['objective_multisample_processes'] = nproc
                self['objective_multisample_processes'] = min(self['objective_multisample_processes'],nproc)
                self['objective_multisample_threads'] = 1
            elif (self['objective_evaluation_parallelism']==True):
                # nproc = max(1,math.floor((computer.cores*computer.nodes)/(self['objective_nprocmax']+1)))
                if(self['objective_multisample_threads'] is None):
                    self['objective_multisample_threads'] = computer.cores
                # self['objective_multisample_threads'] = min(self['objective_multisample_threads'],computer.cores,nproc)
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
                if(self['objective_nospawn']==True):
                    if(self['objective_evaluation_parallelism']==True):
                        ncore_obj = self['objective_multisample_processes']*(self['objective_nprocmax'])+1
                    else:
                        ncore_obj = (self['objective_nprocmax'])
                else:    
                    if(self['objective_evaluation_parallelism']==True):
                        ncore_obj = self['objective_multisample_processes']*(self['objective_nprocmax']+1)+1
                    else:
                        ncore_obj = (self['objective_nprocmax']+1)
            else:
                if(self['objective_nospawn']==True):
                    ncore_obj = self['objective_multisample_threads']*(self['objective_nprocmax'])
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

