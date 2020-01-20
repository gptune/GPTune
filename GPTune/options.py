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
		mpi_comm = None          # The mpi communiator that invokes gptune if mpi4py is installed
		distributed_memory_parallelism = False   # Using distributed_memory_parallelism for the modeling (one MPI per model restart) and search phase (one MPI per task)
		shared_memory_parallelism      = False   # Using shared_memory_parallelism for the modeling (one MPI per model restart) and search phase (one MPI per task)
		constraints_evaluation_parallelism = False  # Reserved option 
		objective_evaluation_parallelism   = False  # Reserved option
		verbose = False     # Control the verbosity level
		oversubscribe = False     # Set this to True when the physical core count is less than computer.nodes*computer.cores and the --oversubscribe MPI runtime option is used 

		
		""" Options for the sampling phase """
		sample_class = 'SampleLHSMDU' # Supported sample classes: 'SampleLHSMDU', 'SampleOpenTURNS'
		sample_algo = 'LHS-MDU' # Supported sample algorithms: 'LHS-MDU' --Latin hypercube sampling with multidimensional uniformity, 'MCS' --Monte Carlo Sampling
		sample_max_iter = 10**9  # Maximum number of iterations for generating random sampeles and testing the constraints

		
		
		""" Options for the modeling phase """
		model_class = 'Model_LCM' # Supported sample algorithms: 'Model_GPy_LCM' -- LCM from GPy, 'Model_LCM' -- LCM with fast and parallel inversion, 'Model_DGP' -- deep Gaussian process
		model_threads = None  # Number of threads used for building one GP model in Model_LCM
		model_processes = None # Number of MPIs used for building one GP model in Model_LCM
		model_groups = 1  # Reserved option
		model_restarts = 1 # Number of random starts each building one initial GP model
		model_restart_processes = None  # Number of MPIs each handling one random start
		model_restart_threads = None   # Number of threads each handling one random start 
		model_max_iters = 15000   # Number of maximum iterations for the optimizers
		model_latent = None # Number of latent functions for building one LCM model, defaults to number of tasks
		model_sparse = False # Whether to use SparseGPRegression or SparseGPCoregionalizedRegression from Model_GPy_LCM
		model_inducing = None # Number of inducing points for SparseGPRegression or SparseGPCoregionalizedRegression
		model_layers = 2 # Number of layers for Model_DGP

		
		""" Options for the search phase """
		search_class = 'SearchPyGMO' # Supported searcher classes: 'SearchPyGMO'
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
		search_best_N = 1  # Maximum number of points selected using a multi-objective search algorithm 

		
		self.update(locals())
		self.update(kwargs)
		self.pop('self')


	def validate(self, computer, **kwargs):		
		
		"""  modify the options as needed """
		if (self['distributed_memory_parallelism'] and self['shared_memory_parallelism']):
			self['shared_memory_parallelism']=False
		
		if (self['distributed_memory_parallelism']):
			if(self['search_multitask_processes'] is None):
				self['search_multitask_processes'] = computer.cores*computer.nodes-1
			self['search_multitask_threads'] = 1
			if(self['model_restart_processes'] is None):
				self['model_restart_processes'] = 1
			self['model_restart_processes'] = min(self['model_restarts'],self['model_restart_processes'])
			self['model_restart_threads'] = 1		
		elif(self['shared_memory_parallelism']): 
			self['search_multitask_processes'] = 1
			if(self['search_multitask_threads'] is None):
				self['search_multitask_threads'] = computer.cores
			self['model_restart_processes'] = 1	
			if(self['model_restart_threads'] is None):
				self['model_restart_threads'] = 1			
			self['model_restart_threads'] = min(min(self['model_restarts'],self['model_restart_threads']),computer.cores)
		else:
			self['search_multitask_processes'] = 1
			self['search_multitask_threads'] = 1
			self['model_restart_processes'] = 1			
			self['model_restart_threads'] = 1		
		
		if (self['model_class']=='Model_LCM'):
			if(self['model_processes'] is None):
				self['model_processes'] = max(1,math.floor((computer.cores*computer.nodes-1)/self['model_restart_processes']/self['model_restart_threads']))
			self['model_threads'] =1
		else:
			self['model_processes'] = 1
			self['model_threads'] = 1  # YL: this requires more thoughts, maybe math.floor((computer.cores)/self['model_restart_threads'])
		
		if(self['search_threads'] is None):
			self['search_threads'] = 1# YL: this requires more thoughts, maybe max(1,math.floor((computer.cores)/self['search_multitask_threads']))	
		
		
		print('\n\n------Validating the options')			
		# print("Parallelism in GTune:")
		
		print("  model_processes:", self['model_processes'])
		print("  model_threads:", self['model_threads'])
		print("  search_processes:", self['search_processes'])
		print("  search_threads:", self['search_threads'])		
		
		print("  distributed_memory_parallelism:", self['distributed_memory_parallelism'])
		print("  shared_memory_parallelism:", self['shared_memory_parallelism'])
		print("  model_restart_processes:", self['model_restart_processes'])
		print("  model_restart_threads:", self['model_restart_threads'])
		print("  search_multitask_processes:", self['search_multitask_processes'])		
		print("  search_multitask_threads:", self['search_multitask_threads'])		
		
		if(self['oversubscribe']==False):
			if (computer.cores*computer.nodes<=1):
				raise Exception("the computer should has at least 2 total cores")
					
			if ((computer.cores*computer.nodes-1)<self['model_restart_processes']*self['model_restart_threads']*self['model_processes']*self['model_threads']):
				raise Exception("model_restart_processes*model_restart_threads*model_processes*model_threads should not exceed cores*nodes-1")
			if (self['distributed_memory_parallelism'] and (computer.cores*computer.nodes-1)<self['search_multitask_processes']*self['search_multitask_threads']*self['search_processes']*self['search_threads']):
				raise Exception("search_multitask_processes*search_multitask_threads*search_processes*search_threads should not exceed cores*nodes-1")	
				
				
