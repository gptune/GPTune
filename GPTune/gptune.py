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

import copy
import functools

from autotune.problem import TuningProblem

from problem import Problem
from computer import Computer
from data import Data
from options import Options
from sample import *
from model import *
from search import *

import mpi4py
from mpi4py import MPI		  
class GPTune(object):

	def __init__(self, tuningproblem : TuningProblem, computer : Computer = None, data : Data = None, options : Options = None, **kwargs):

		"""
		tuningproblem: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
		computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
		data         : object containing the data of a previous tuning (See file 'GPTune/data.py')
		options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
		"""

		self.problem  = Problem(tuningproblem)
		if (computer is None):
			computer = Computer()
		self.computer = computer
		if (data is None):
			data = Data(self.problem)
		self.data     = data
		if (options is None):
			options = Options()
		self.options  = options

		if (self.options['distributed_memory_parallelism']\
			and\
			('mpi4py' in sys.modules)): # make sure that the mpi4py has been loaded successfully
		#            if ('mpi_comm' in kwargs):
		#                self.mpi_comm = kwargs['mpi_comm']
			if (options['mpi_comm'] is not None):
				self.mpi_comm = options['mpi_comm']
			else:
				self.mpi_comm = mpi4py.MPI.COMM_WORLD
		#            self.mpi_rank = self.mpi_comm.Get_rank()
		#            self.mpi_size = self.mpi_comm.Get_size()
		else: # fall back to sequential tuning (MPI wise, but still multithreaded)
			self.mpi_comm = None
		#            self.mpi_rank = 0
		#            self.mpi_size = 1

	def MLA(self, NS, NS1 = None, NI = None, **kwargs):

		print('\n\n\n------Starting MLA with %d tasks '%(self.problem.DI))	

		kwargs = copy.deepcopy(self.options)
		kwargs.update(kwargs)
		kwargs.update({'mpi_comm' : self.mpi_comm})

		""" Multi-task Learning Autotuning """

########## normalize the data as the user always work in the original space
		if self.data.T is not None:
			tmp=[]
			for t in self.data.T:		
				tNorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
				tmp.append(tNorm)		
			self.data.T=tmp
		if self.data.X is not None:			
			tmp=[]
			for x in self.data.X:		
				xNorm = self.problem.PS.transform(np.array(x, ndmin=2))[0]
				tmp.append(xNorm)		
			self.data.X=tmp		

		
		
#        if (self.mpi_rank == 0):

		sampler = eval(f'{kwargs["sample_class"]}()')
		if (self.data.T is None):

			if (NI is None):
				raise Exception("Number of problems to be generated (NI) is not defined")

			check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = True, kwargs = kwargs)
			self.data.T = sampler.sample_inputs(n_samples = NI, IS = self.problem.IS, check_constraints = check_constraints, **kwargs)
			# print("riji",self.data.T)
		if (self.data.X is None):

			if (NS1 is None):
				NS1 = min(NS - 1, 3 * self.problem.DP) # General heuristic rule in the litterature

			check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
			self.data.X = sampler.sample_parameters(n_samples = NS1, T = self.data.T, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
#            #XXX add the info of problem.models here
#            for X2 in X:
#                for x in X2:
#                    x = np.concatenate(x, np.array([m(x) for m in self.problems.models]))
		# print("good?")
		if (self.data.Y is None):
			self.data.Y = self.computer.evaluate_objective(self.problem, self.data.T, self.data.X, kwargs = kwargs) 
		# print("good!")	
#            if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
#                mpi_comm.bcast(self.data, root=0)
#
#        else:
#
#            self.data = mpi_comm.bcast(None, root=0)

		NS2 = NS - len(self.data.X[0])

		modeler  = eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')
		searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')
		for optiter in range(NS2): # YL: each iteration adds one sample until total #sample reaches NS

			newdata = Data(problem = self.problem, T = self.data.T)
			# print("before train",optiter,NS2)
			modeler.train(data = self.data, **kwargs)
			# print("after train",self.data.X,'d',newdata.X) 
			# print("after train",self.data.Y,'d',newdata.Y) 
			# print("after train",self.data.T,'d',newdata.T) 
			res = searcher.search_multitask(data = self.data, model = modeler, **kwargs)
			newdata.X = [x[1][0] for x in res]
	#XXX add the info of problem.models here

	#            if (self.mpi_rank == 0):

			newdata.Y = self.computer.evaluate_objective(problem = self.problem, fun = self.problem.objective, T = newdata.T, X = newdata.X, kwargs = kwargs)

	#                if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
	#                    mpi_comm.bcast(newdata.Y, root=0)
	#
	#            else:
	#
	#                newdata.Y = mpi_comm.bcast(None, root=0)

			self.data.merge(newdata)
			
				
########## denormalize the data as the user always work in the original space
		if self.data.T is not None:        
			tmp=[]
			for t in self.data.T:		
				tOrig = self.problem.IS.inverse_transform(np.array(t, ndmin=2))[0]
				tmp.append(tOrig)		
			self.data.T=tmp
		if self.data.X is not None:        
			tmp=[]
			for x in self.data.X:		
				xOrig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))
				tmp.append(xOrig)		
			self.data.X=tmp		
			
			
		return (copy.deepcopy(self.data), modeler)


	def TLA1(self, Tnew, nruns):
       
		print('\n\n\n------Starting TLA1 for task: ',Tnew)
		# Initialization
		kwargs = copy.deepcopy(self.options)
		ntso = len(self.data.T)
		ntsn = len(Tnew)

		XSopt =[]
		for i in range(ntso):
			XSopt.append(self.data.X[i][np.argmin(self.data.Y[i])])	
		# YSopt = np.array([[self.data.Y[k].min()] for k in range(ntso)])
		MSopt = []



		# convert the task spaces to the normalized spaces
		tNorms=[]
		for t in self.data.T:		
			tNorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
			tNorms.append(tNorm.reshape((-1, self.problem.DI)))		
		tNorms = np.vstack([tNorms[i] for i in range(ntso)]).reshape((ntso,self.problem.DI))
  
		tmp=[]
		for t in Tnew:		
			tNorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
			tmp.append(tNorm.reshape((-1, self.problem.DI)))		
		TnewNorms=np.vstack([tmp[i] for i in range(ntsn)]).reshape((ntsn,self.problem.DI))
   
  
  
		# convert the parameter spaces to the normalized spaces  
		XSoptNorms = self.problem.PS.transform(XSopt)
		columns = []
		for j in range(self.problem.DP):
			columns.append([])
		for i in range(ntso):
			for j in range(self.problem.DP):
				columns[j].append(XSoptNorms[i][j])
		XSoptNorms = []
		for j in range(self.problem.DP):
			XSoptNorms.append(np.asarray(columns[j]).reshape((ntso, -1))) 

		

		# Predict optimums of new tasks
		for k in range(self.problem.DP):
			K = GPy.kern.RBF(input_dim=self.problem.DI)
			M = GPy.models.GPRegression(tNorms, XSoptNorms[k], K)
			# M.optimize_restarts(num_restarts = 10, robust=True, verbose=False, parallel=False, num_processes=None, messages="False")
			M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust=True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = "False", optimizer = 'lbfgs', start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
			MSopt.append(M)

		aprxoptsNorm=np.hstack([MSopt[k].predict_noiseless(TnewNorms)[0] for k in range(self.problem.DP)])  # the index [0] is the mean value, [1] is the variance
		aprxoptsNorm=np.minimum(aprxoptsNorm,(1-1e-12)*np.ones((ntsn,self.problem.DP)))
		aprxoptsNorm=np.maximum(aprxoptsNorm,(1e-12)*np.ones((ntsn,self.problem.DP)))
		# print('aprxoptsNorm',aprxoptsNorm,type(aprxoptsNorm))
		aprxopts = self.problem.PS.inverse_transform(aprxoptsNorm)
		# print('aprxopts',aprxopts,type(aprxopts),type(aprxopts[0]))
		
  
		aprxoptsNormList=[]
		# TnewNormList=[]
		for i in range(ntsn):
			aprxoptsNormList.append([aprxoptsNorm[i,:]])  # this makes sure for each task, there is only one sample parameter set
			# TnewNormList.append(TnewNorms[i,:])

		Y = self.computer.evaluate_objective(problem = self.problem, fun = self.problem.objective, T = TnewNorms, X =aprxoptsNormList, kwargs = kwargs)
  

		#        print(aprxopts)
		#        pickle.dump(aprxopts, open('TLA1.pkl', 'w'))

		return (aprxopts,Y)

	def TLA2(): # co-Kriging

		pass

