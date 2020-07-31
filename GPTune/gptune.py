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
import time

from autotune.problem import TuningProblem

from problem import Problem
from computer import Computer
from data import Data
from options import Options
from sample import *
from model import *
from search import *
import math

import mpi4py
from mpi4py import MPI		  
import numpy as np

class GPTune(object):

	def __init__(self, tuningproblem : TuningProblem, computer : Computer = None, data : Data = None, options : Options = None, driverabspath=None, models_update=None, **kwargs):

		"""
		tuningproblem: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
		computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
		data         : object containing the data of a previous tuning (See file 'GPTune/data.py')
		options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
		"""
		self.problem  = Problem(tuningproblem,driverabspath=driverabspath,models_update=models_update)
		if (computer is None):
			computer = Computer()
		self.computer = computer
		if (data is None):
			data = Data(self.problem)
		self.data     = data
		if (options is None):
			options = Options()
		self.options  = options

	def MLA(self, NS, NS1 = None, NI = None, Igiven = None, **kwargs):

		print('\n\n\n------Starting MLA with %d tasks and %d samples each '%(NI,NS))	
		stats = {
			"time_total": 0,
			"time_sample_init": 0,			
			"time_fun": 0,
			"time_search": 0,
			"time_model": 0
		}
		time_fun=0
		time_sample_init=0
		time_search=0
		time_model=0
				
		np.set_printoptions(suppress=False,precision=4)
		
		if (self.data.P is not None and len(self.data.P[0])>=NS):
			print('self.data.P[0])>=NS, no need to run MLA. Returning...')
			return (copy.deepcopy(self.data), None,stats)	
		
		t3 = time.time_ns()
		
		t1 = time.time_ns()	

		options1 = copy.deepcopy(self.options)
		kwargs.update(options1)

		""" Multi-task Learning Autotuning """

		
		if(Igiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
			self.data.I = Igiven 

########## normalize the data as the user always work in the original space

		if self.data.I is not None: # from a list of lists to a 2D numpy array
			self.data.I = self.problem.IS.transform(self.data.I)

		if self.data.P is not None:	# from a list of (list of lists) to a list of 2D numpy arrays		
			tmp=[]
			for x in self.data.P:		
				xNorm = self.problem.PS.transform(x)
				tmp.append(xNorm)
			self.data.P=tmp				
		
#        if (self.mpi_rank == 0):

		sampler = eval(f'{kwargs["sample_class"]}()')
		if (self.data.I is None):

			if (NI is None):
				raise Exception("Number of problems to be generated (NI) is not defined")

			check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = True, kwargs = kwargs)
			self.data.I = sampler.sample_inputs(n_samples = NI, IS = self.problem.IS, check_constraints = check_constraints, **kwargs)
			# print("riji",type(self.data.I),type(self.data.I[0]))
			self.data.D = [{}] * NI
		else:
			if (self.data.D is None):
				self.data.D = [{}] * NI	
		
		if (self.data.P is not None and len(self.data.P) !=len(self.data.I)):
			raise Exception("len(self.data.P) !=len(self.data.I)")		
		
		if (self.data.P is None):
			if (NS1 is not None and NS1>NS):
				raise Exception("NS1>NS")
				
			if (NS1 is None):
				NS1 = min(NS - 1, 3 * self.problem.DP) # General heuristic rule in the litterature

			check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
			self.data.P = sampler.sample_parameters(n_samples = NS1, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
#            #XXX add the info of problem.models here
#            for P2 in P:
#                for x in P2:
#                    x = np.concatenate(x, np.array([m(x) for m in self.problems.models]))
		# print("good?")
		
		if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
			raise Exception("len(self.data.O) !=len(self.data.I)")
		
		t2 = time.time_ns()	
		time_sample_init = time_sample_init	+ (t2-t1)/1e9	

		t1 = time.time_ns()
		if (self.data.O is None):
			self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, options = kwargs) 
		t2 = time.time_ns()
		time_fun = time_fun + (t2-t1)/1e9
		# print(self.data.O)
		# print("good!")	
#            if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
#                mpi_comm.bcast(self.data, root=0)
#
#        else:
#
#            self.data = mpi_comm.bcast(None, root=0)
		# mpi4py.MPI.COMM_WORLD.Barrier()
		modelers  = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
		searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer)')
		optiter = 0
		while len(self.data.P[0])<NS:# YL: each iteration adds 1 (if single objective) or at most kwargs["search_more_samples"] (if multi-objective) sample until total #sample reaches NS
		# for optiter in range(NS - len(self.data.P[0])): 

			if(self.problem.models_update is not None):
				########## denormalize the data as the user always work in the original space
				tmpdata = copy.deepcopy(self.data)
				if tmpdata.I is not None:    # from 2D numpy array to a list of lists    
					tmpdata.I = self.problem.IS.inverse_transform(tmpdata.I)
				if tmpdata.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)       
					tmp=[]
					for x in tmpdata.P:		
						xOrig = self.problem.PS.inverse_transform(x)
						tmp.append(xOrig)		
					tmpdata.P=tmp		
				self.problem.models_update(tmpdata)
				self.data.D = tmpdata.D

			# print("riji",type(self.data.I),type(self.data.I[0]))
			newdata = Data(problem = self.problem, I = self.data.I, D = self.data.D)
			print("MLA iteration: ",optiter)
			optiter = optiter + 1
			t1 = time.time_ns()
			for o in range(self.problem.DO):
				tmpdata = copy.deepcopy(self.data)
				tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]
				if(self.problem.models is not None):
					for i in range(len(tmpdata.P)):
						points0 = tmpdata.D[i]
						t = tmpdata.I[i]
						I_orig = self.problem.IS.inverse_transform(np.array(t, ndmin=2))[0]					
						points1 = {self.problem.IS[k].name: I_orig[k] for k in range(self.problem.DI)}
						modeldata=[]
						for p in range(len(tmpdata.P[i])):
							x = tmpdata.P[i][p]
							x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]		
							points = {self.problem.PS[k].name: x_orig[k] for k in range(self.problem.DP)}
							points.update(points1)
							points.update(points0)
							modeldata.append(self.problem.models(points))
						modeldata=np.array(modeldata)	
						tmpdata.P[i] = np.hstack((tmpdata.P[i],modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space 
				# print(tmpdata.P[0])
				modelers[o].train(data = tmpdata, **kwargs)
			
			t2 = time.time_ns()
			time_model = time_model + (t2-t1)/1e9
		
			t1 = time.time_ns()
			res = searcher.search_multitask(data = self.data, models = modelers, **kwargs)
			
			more_samples=NS-len(self.data.P[0]) # YL: this makes sure P has the same length across all tasks
			for x in res:
				more_samples=min(more_samples,x[1][0].shape[0])	
			newdata.P = [x[1][0][0:more_samples,:] for x in res]
			# print(more_samples,newdata.P)
			t2 = time.time_ns()
			time_search = time_search + (t2-t1)/1e9		
	#XXX add the info of problem.models here

	#            if (self.mpi_rank == 0):

			t1 = time.time_ns()
			newdata.O = self.computer.evaluate_objective(problem = self.problem, I = newdata.I, P = newdata.P, D = newdata.D, options = kwargs)
			t2 = time.time_ns()
			time_fun = time_fun + (t2-t1)/1e9		
	#                if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
	#                    mpi_comm.bcast(newdata.O, root=0)
	#
	#            else:
	#
	#                newdata.O = mpi_comm.bcast(None, root=0)	
			self.data.merge(newdata)
		
########## denormalize the data as the user always work in the original space
		if self.data.I is not None:    # from 2D numpy array to a list of lists    
			self.data.I = self.problem.IS.inverse_transform(self.data.I)
		if self.data.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)       
			tmp=[]
			for x in self.data.P:		
				xOrig = self.problem.PS.inverse_transform(x)
				tmp.append(xOrig)		
			self.data.P=tmp		

		t4 = time.time_ns()
		stats['time_total'] = (t4-t3)/1e9		
		stats['time_fun'] = time_fun			
		stats['time_model'] = time_model			
		stats['time_search'] = time_search			
		stats['time_sample_init'] = time_sample_init			
		
		
		return (copy.deepcopy(self.data), modelers,stats)

		
	def TLA1(self, Tnew, NS):

		print('\n\n\n------Starting TLA1 for task: ',Tnew)

		stats = {
			"time_total": 0,
			"time_fun": 0
		}
		time_fun=0
		
		t3=time.time_ns()
		# Initialization
		kwargs = copy.deepcopy(self.options)
		ntso = len(self.data.I)
		ntsn = len(Tnew)

		if(self.data.O[0].shape[1]>1):
			raise Exception("TLA1 only works for single-objective tuning")
		
		PSopt =[]
		for i in range(ntso):
			PSopt.append(self.data.P[i][np.argmin(self.data.O[i])])	
		# YSopt = np.array([[self.data.O[k].min()] for k in range(ntso)])
		MSopt = []



		# convert the task spaces to the normalized spaces
		INorms=[]
		for t in self.data.I:		
			INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
			INorms.append(INorm.reshape((-1, self.problem.DI)))		
		INorms = np.vstack([INorms[i] for i in range(ntso)]).reshape((ntso,self.problem.DI))

		tmp=[]
		for t in Tnew:		
			INorm = self.problem.IS.transform(np.array(t, ndmin=2))[0]
			tmp.append(INorm.reshape((-1, self.problem.DI)))		
		InewNorms=np.vstack([tmp[i] for i in range(ntsn)]).reshape((ntsn,self.problem.DI))

		# convert the parameter spaces to the normalized spaces  
		PSoptNorms = self.problem.PS.transform(PSopt)
		columns = []
		for j in range(self.problem.DP):
			columns.append([])
		for i in range(ntso):
			for j in range(self.problem.DP):
				columns[j].append(PSoptNorms[i][j])
		PSoptNorms = []
		for j in range(self.problem.DP):
			PSoptNorms.append(np.asarray(columns[j]).reshape((ntso, -1))) 

		

		# Predict optimums of new tasks
		for k in range(self.problem.DP):
			K = GPy.kern.RBF(input_dim=self.problem.DI)
			M = GPy.models.GPRegression(INorms, PSoptNorms[k], K)
			# M.optimize_restarts(num_restarts = 10, robust=True, verbose=False, parallel=False, num_processes=None, messages="False")
			M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust=True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = 'lbfgs', start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
			MSopt.append(M)

		aprxoptsNorm=np.hstack([MSopt[k].predict_noiseless(InewNorms)[0] for k in range(self.problem.DP)])  # the index [0] is the mean value, [1] is the variance
		aprxoptsNorm=np.minimum(aprxoptsNorm,(1-1e-12)*np.ones((ntsn,self.problem.DP)))
		aprxoptsNorm=np.maximum(aprxoptsNorm,(1e-12)*np.ones((ntsn,self.problem.DP)))
		# print('aprxoptsNorm',aprxoptsNorm,type(aprxoptsNorm))
		aprxopts = self.problem.PS.inverse_transform(aprxoptsNorm)
		# print('aprxopts',aprxopts,type(aprxopts),type(aprxopts[0]))
		

		aprxoptsNormList=[]
		# TnewNormList=[]
		for i in range(ntsn):
			aprxoptsNormList.append([aprxoptsNorm[i,:]])  # this makes sure for each task, there is only one sample parameter set
			# InewNormList.append(InewNorms[i,:])
		
		t1 = time.time_ns()
		O = self.computer.evaluate_objective(problem = self.problem, I = InewNorms, P =aprxoptsNormList, options = kwargs)
		t2 = time.time_ns()
		time_fun = time_fun + (t2-t1)/1e9		

		#        print(aprxopts)
		#        pickle.dump(aprxopts, open('TLA1.pkl', 'w'))

		t4 = time.time_ns()
		stats['time_total'] = (t4-t3)/1e9		
		stats['time_fun'] = time_fun		
		
		return (aprxopts,O,stats)

	def TLA2(): # co-Kriging

		pass


class GPTune_MB(object):

	def __init__(self, tp : TuningProblem, computer : Computer = None, options : Options = None, **kwargs):

		"""
		tp: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
		computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
		options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
		"""

		smax = int(np.floor(np.log10(options['budget_max']/options['budget_min'])/np.log10(options['budget_base'])))
		self.budgets=[options['budget_max']/options['budget_base']**x for x in range(smax+1)]
		# print(self.budgets)

		parameter_space = tp.parameter_space
		output_space = tp.output_space
		objectives = tp.objective
		constraints = tp.constraints

		""" insert "budget" as the first dimension of the input space """
		inputs = [Real     (options['budget_min']-1e-12, options['budget_max'], transform="normalize", name="budget")]

		for n,p in zip(tp.input_space.dimension_names,tp.input_space.dimensions):
			if (isinstance(p, Real)):
				inputs.append(Real(p.bounds[0], p.bounds[1], transform="normalize", name=n))
			elif (isinstance(p, Integer)):
				inputs.append(Integer(p.bounds[0], p.bounds[1], transform="normalize", name=n))
			elif (isinstance(p, Categorical)):
				inputs.append(Categoricalnorm (list(p.bounds), transform="onehot", name=n))
			else:
				raise Exception("Unknown parameter type")

		# print(inputs)
		input_space = Space(inputs)
		
		self.tp = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None)
		self.computer = computer
		self.options  = options
		self.data 	  = Data(tp)


	def MB_LCM(self, NS = None, Igiven = None, **kwargs):
		"""
		Igiven		 : a list of tasks 
		NS			 : number of samples in the highest budget arm
		"""

		np.set_printoptions(suppress=False,precision=4)
		print('\n\n\n------Starting MB_LCM (multi-arm bandit with LCM) with %d samples for task'%(NS),Igiven)	

		stats = {
			"time_total": 0,
			"time_sample_init": 0,			
			"time_fun": 0,
			"time_search": 0,
			"time_model": 0
		}
		time_fun=0
		time_sample_init=0
		time_search=0
		time_model=0

		self.NSs=[int(self.options['budget_max']/x*NS) for x in self.budgets]
		info = [[x,y] for x,y in zip(self.budgets,self.NSs)]
		print('total samples:',info)
		
		data = Data(self.tp)   # having the budgets not fully sampled before SH
		data1 = Data(self.tp)  # having the budgets fully sampled before SH
		data1.I=[]
		data1.P=[]
		data1.O=[]
		data1.D=[]

		for s in range(len(self.budgets)): # loop over the budget levels
			budget = self.budgets[s]
			ns = self.NSs[s]
			newtasks=[]
			for s1 in range(s,len(self.budgets)): 
				for t in range(len(Igiven)):
					budget1 = self.budgets[s1]
					tmp = [budget1]+Igiven[t]
					newtasks.append(tmp)

			gt = GPTune(self.tp, computer=self.computer, data=data, options=self.options)
			(data, modeler, stats0) = gt.MLA(NS=ns, Igiven=newtasks, NI=len(newtasks), NS1=int(ns/2))
			data1.I += data.I[0:len(Igiven)]
			data1.P += data.P[0:len(Igiven)]
			data1.O += data.O[0:len(Igiven)]
			data1.D += data.D[0:len(Igiven)]
			del data.I[0:len(Igiven)]
			del data.P[0:len(Igiven)]
			del data.O[0:len(Igiven)]
			del data.D[0:len(Igiven)]


			stats['time_total'] += stats0['time_total']	
			stats['time_fun'] += stats0['time_fun']
			stats['time_model'] += stats0['time_model']
			stats['time_search'] += stats0['time_search']
			stats['time_sample_init'] += stats0['time_sample_init']	

		# print(data1.I)
		# print(data1.P)
		# print(data1.O)
		self.data.I = Igiven
		self.data.P = data1.P[0:len(Igiven)]  # this will be updated by SH
		self.data.O = data1.O[0:len(Igiven)]  # this will be updated by SH
		#todo SH on each arm and return all samples of the highest fidelity in self.data

		return (copy.deepcopy(self.data), stats)

		

