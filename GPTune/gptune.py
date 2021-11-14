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
from autotune.space import *

from problem import Problem
from computer import Computer
from options import Options
from data import *
from database import *
from sample import *
from model import *
from search import *
import math
import os

import mpi4py
import numpy as np

import json
from filelock import Timeout, FileLock

class GPTune(object):

    def __init__(self, tuningproblem : TuningProblem, computer : Computer = None, data : Data = None, historydb : HistoryDB = None, options : Options = None, driverabspath=None, models_update=None, **kwargs):

        """
        tuningproblem: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
        computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
        data         : object containing the data of a previous tuning (See file 'GPTune/data.py')
        options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
        """
        self.tuningproblem = tuningproblem
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
        if (historydb is None):
            historydb = HistoryDB()
        self.historydb = historydb

    def LoadSurrogateModel(self, model_data : dict, **kwargs):

        Igiven = model_data["task_parameters"]

        """ Load history function evaluation data """

        self.historydb.load_model_func_eval(data = self.data, problem = self.problem,
                Igiven = Igiven, model_data = model_data)

        """ Update data space """

        if(Igiven is not None and self.data.I is None):
            self.data.I = Igiven

        # normalize the data as the user always work in the original space
        if self.data.P is not None:
            tmp=[]
            for x in self.data.P:
                xNorm = self.problem.PS.transform(x)
                tmp.append(xNorm)
            self.data.P=tmp
        if self.data.I is not None:
            self.data.I = self.problem.IS.transform(self.data.I)

        if (self.data.O is None and self.data.P is not None and self.data.I is not None):
            self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, options = kwargs)

        if (self.data.D is None):
            self.data.D = [{}] * len(self.data.I)

        if (self.data.P is not None and len(self.data.P) !=len(self.data.I)):
            raise Exception("len(self.data.P) !=len(self.data.I)")

        if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
            raise Exception("len(self.data.O) !=len(self.data.I)")

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Reproduce surrogate models """

        modelers = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO

        for i in range(self.problem.DO):
            modelers[i].gen_model_from_hyperparameters(self.data,
                    model_data["hyperparameters"],
                    **kwargs)

        def model_function(point):

            task_parameter_names = [self.problem.IS[k].name for k in range(len(self.problem.IS))]
            tuning_parameter_names = [self.problem.PS[k].name for k in range(len(self.problem.PS))]
            tuning_parameter_types = [type(self.problem.PS[k]).__name__ for k in range(len(self.problem.PS))]
            output_names = [self.problem.OS[k].name for k in range(len(self.problem.OS))]

            #print ("task_parameter_names: ", task_parameter_names)
            #print ("tuning_parameter_names: ", tuning_parameter_names)
            #print ("tuning_parameter_types: ", tuning_parameter_types)
            #print ("output_names: ", output_names)
            #print ("Igiven: ", Igiven)

            input_task = []
            for task_parameter_name in task_parameter_names:
                input_task.append(point[task_parameter_name])

            tid = -1
            for i in range(len(Igiven)):
                is_equal = True
                for j in range(len(Igiven[i])):
                    if Igiven[i][j] != input_task[j]:
                        is_equal = False
                if (is_equal):
                    tid = i
                    break
            if tid == -1:
                print ("[Error] cannot find model for the given input task: ", input_task)
                return None

            bound_checked = True

            input_tuning_parameters = []
            for tuning_parameter_name in tuning_parameter_names:
                if tuning_parameter_types[tuning_parameter_names.index(tuning_parameter_name)] == "Integer" or\
                   tuning_parameter_types[tuning_parameter_names.index(tuning_parameter_name)] == "Integer":
                    lower_bound, upper_bound = self.problem.PS.bounds[tuning_parameter_names.index(tuning_parameter_name)]
                    if point[tuning_parameter_name] < lower_bound or point[tuning_parameter_name] > upper_bound:
                        bound_checked = False
                input_tuning_parameters.append(point[tuning_parameter_name])

            #tuning_parameter_names_model_order = parameter_names
            #print ("tuning_parameter_names_model_order")
            #print (tuning_parameter_names_model_order)
            #input_tuning_parameters_transformed_reordered = []
            #for parameter_name in tuning_parameter_names_model_order:
            #    print ("parameter_name: ", parameter_name)
            #    print ("index: ", tuning_parameter_names.index(parameter_name))
            #    print ("val: ", input_tuning_parameters_transformed[tuning_parameter_names.index(parameter_name)])
            #    input_tuning_parameters_transformed_reordered.append(
            #            input_tuning_parameters_transformed[tuning_parameter_names.index(parameter_name)])
            #print ("input_tuning_parameters_transformed_reordered")
            #print (input_tuning_parameters_transformed_reordered)

            ret = {}

            if (bound_checked == True):
                input_tuning_parameters_transformed = self.problem.PS.transform([input_tuning_parameters])[0]

                for o in range(self.problem.DO):
                    (mu, var) = modelers[o].predict(np.array(input_tuning_parameters_transformed),tid)

                    ret[output_names[o]] = np.array(mu).tolist()
                    ret[output_names[o]+"_var"] = var
            else:
                for o in range(self.problem.DO):
                    # TODO: the value depends on the problem (output) definition
                    (mu, var) = [[1000000.0]], 0
                    ret[output_names[o]] = np.array(mu).tolist()
                    ret[output_names[o]+"_var"] = var

            ret["source"] = "model_function"

            return ret

        return (modelers, model_function)

    def MLA_LoadModel(self, NS = 0, Igiven = None, method = "max_evals", update = 0, model_uids = None, **kwargs):
        print('\n\n\n------Starting MLA with Trained Model for %d tasks and %d samples each '%(len(Igiven),NS))
        stats = {
            "time_total": 0,
            "time_fun": 0,
            "time_search": 0,
            "time_model": 0,
            "modeling_time":[],
            "modeling_iteration":[]
        }
        time_fun=0
        time_search=0
        time_model=0

        """ Load history function evaluation data """
        self.historydb.load_history_func_eval(self.data, self.problem, Igiven)
        np.set_printoptions(suppress=False,precision=4)
        NSmin=0
        if (self.data.P is not None):
            NSmin = min(map(len, self.data.P)) # the number of samples per task in existing tuning data can be different

        #if (self.data.P is not None and NSmin>=NS and self.data.O is not None):
        #    print('NSmin>=NS, no need to run MLA. Returning...')
        #    return (copy.deepcopy(self.data), None,stats)
        """ Set (additional) number of samples for autotuning """
        NS = NSmin + NS

        t3 = time.time_ns()

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Multi-task Learning Autotuning """
        if(Igiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Igiven

        # normalize the data as the user always work in the original space
        if self.data.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
            tmp=[]
            for x in self.data.P:
                xNorm = self.problem.PS.transform(x)
                tmp.append(xNorm)
            self.data.P=tmp
        if self.data.I is not None: # from a list of lists to a 2D numpy array
            self.data.I = self.problem.IS.transform(self.data.I)

        if (self.data.O is None and self.data.P is not None and self.data.I is not None): # tuning parameters and task parameters are given, but the output is none
            self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, options = kwargs)

        if (self.data.D is None):
            self.data.D = [{}] * len(self.data.I) #NI

        if (self.data.P is not None and len(self.data.P) !=len(self.data.I)):
            raise Exception("len(self.data.P) !=len(self.data.I)")

        if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
            raise Exception("len(self.data.O) !=len(self.data.I)")

        modelers  = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
        for i in range(self.problem.DO):
            # current limitations
            # - only model LCM
            # - not considering edge cases (e.g. no model is available)
            if model_uids == None:
                #TODO CHECK: make self.data is correct (we may need to load (or double check) func eval data based on the model data)

                if method == "max_evals":
                    (hyperparameters, parameter_names) = self.historydb.load_max_evals_surrogate_model_hyperparameters(
                            self.tuningproblem, Igiven, i, kwargs["model_class"])
                elif method == "MLE" or method == "mle":
                    (hyperparameters, parameter_names) = self.historydb.load_MLE_surrogate_model_hyperparameters(
                            self.tuningproblem, Igiven, i, kwargs["model_class"])
                elif method == "AIC" or method == "aic":
                    (hyperparameters, parameter_names) = self.historydb.load_AIC_surrogate_model_hyperparameters(
                            self.tuningproblem, Igiven, i, kwargs["model_class"])
                elif method == "BIC" or method == "bic":
                    (hyperparameters, parameter_names) = self.historydb.load_BIC_model_hyperparameters(
                            self.tuningproblem, Igiven, i, kwargs["model_class"])
                else:
                    (hyperparameters, parameter_names) = self.historydb.load_max_evals_surrogate_model_hyperparameters(
                            self.tuningproblem, Igiven, i, kwargs["model_class"])

            else:
                (hyperparameters, parameter_names) = self.historydb.load_surrogate_model_hyperparameters_by_uid(model_uids[i])
            modelers[i].gen_model_from_hyperparameters(self.data,
                    hyperparameters,
                    **kwargs)

        searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options)')
        model_reupdate = 0
        if update == -1: # search one sample without updating model; then search next with updating model.
            model_reupdate = -1
            update = 1
        optiter = 0
        NSmin = min(map(len, self.data.P))
        while NSmin<NS:# YL: each iteration adds 1 (if single objective) or at most kwargs["search_more_samples"] (if multi-objective) sample until total #sample reaches NS

            if(self.problem.models_update is not None):
                ########## denormalize the data as the user always work in the original space
                tmpdata = copy.deepcopy(self.data)
                if tmpdata.I is not None:    # from 2D numpy array to a list of lists
                    tmpdata.I = self.problem.IS.inverse_transform(tmpdata.I)
                if tmpdata.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
                    tmpdata.P = [self.problem.PS.inverse_transform(x) for x in tmpdata.P]
                self.problem.models_update(tmpdata)
                self.data.D = tmpdata.D

            newdata = Data(problem = self.problem, I = self.data.I, D = self.data.D)
            print("MLA iteration: ",optiter)
            stats["modeling_iteration"].append(0)
            optiter = optiter + 1
            model_reupdate = model_reupdate + 1
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
                for i in range(len(tmpdata.P)):   # LCM requires the same number of samples per task, so use the first NSmin samples
                    tmpdata.O[i] = tmpdata.O[i][0:NSmin,:]
                    tmpdata.P[i] = tmpdata.P[i][0:NSmin,:]

                if model_reupdate == update:
                    # print(tmpdata.P[0])
                    #print ("[bestxopt]: len: " + str(len(bestxopt)) + " val: " + str(bestxopt))
                    if (kwargs["model_class"] == "Model_LCM"):
                        (bestxopt, neg_log_marginal_likelihood,
                                gradients, iteration) = \
                            modelers[o].train(data = tmpdata, **kwargs)
                        self.historydb.store_model_LCM(
                                o,
                                self.problem,
                                self.data.I,
                                bestxopt,
                                neg_log_marginal_likelihood,
                                gradients,
                                iteration)
                        stats["modeling_iteration"][optiter-1] += iteration
                    else:
                        modelers[o].train(data = tmpdata, **kwargs)
                        stats["modeling_iteration"][optiter-1] += 0
                    model_reupdate = 0
                else:
                    if (kwargs["model_class"] == "Model_LCM"):
                        stats["modeling_iteration"][optiter-1] += 0

            t2 = time.time_ns()
            stats["modeling_time"].append((t2-t1)/1e9)
            time_model = time_model + (t2-t1)/1e9

            t1 = time.time_ns()
            res = searcher.search_multitask(data = self.data, models = modelers, **kwargs)

            newdata.P = [x[1][0] for x in res]
            for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
                NSi = self.data.P[i].shape[0]
                newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]
            # print(more_samples,newdata.P)
            t2 = time.time_ns()
            time_search = time_search + (t2-t1)/1e9

            t1 = time.time_ns()
            newdata.O = self.computer.evaluate_objective(problem = self.problem,
                    I = newdata.I,
                    P = newdata.P,
                    D = newdata.D,
                    history_db = self.historydb,
                    options = kwargs)
            t2 = time.time_ns()
            time_fun = time_fun + (t2-t1)/1e9
            self.data.merge(newdata)
            NSmin = min(map(len, self.data.P))

        # denormalize the data as the user always work in the original space
        if self.data.I is not None:    # from 2D numpy array to a list of lists
            self.data.I = self.problem.IS.inverse_transform(self.data.I)
        if self.data.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
            self.data.P = [self.problem.PS.inverse_transform(x) for x in self.data.P]

        t4 = time.time_ns()
        stats['time_total'] = (t4-t3)/1e9
        stats['time_fun'] = time_fun
        stats['time_model'] = time_model
        stats['time_search'] = time_search

        return (copy.deepcopy(self.data), modelers, stats)

    def MLA_HistoryDB(self, NS, NS1 = None, NI = None, Igiven = None, **kwargs):
        print('\n\n\n------Starting MLA with HistoryDB with %d tasks and %d samples each '%(NI,NS))
        stats = {
            "time_total": 0,
            "time_sample_init": 0,
            "time_fun": 0,
            "time_search": 0,
            "time_model": 0,
            "func_eval_time":[],
            "search_time":[],
            "modeling_time":[],
            "modeling_iteration":[]
        }
        time_fun=0
        time_sample_init=0
        time_search=0
        time_model=0

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            self.historydb.load_history_func_eval(self.data, self.problem, Igiven)

        np.set_printoptions(suppress=False,precision=4)
        NSmin=0
        NSmax=0
        if (self.data.P is not None):
            NSmin = min(map(len, self.data.P)) # the number of samples per task in existing tuning data can be different
            NSmax = max(map(len, self.data.P)) 

        # """ Set (additional) number of samples for autotuning """
        # NS = NSmin + NS
        if(NSmax>0):
            if (self.data.P is not None and NSmin>=NS and self.data.O is not None):
                print('\nexisting data has at least NSmin=%d samples per task, which is no less than NS=%d, no need to run MLA. Returning...\n'%(NSmin,NS))
                return (copy.deepcopy(self.data), None,stats)
            else:            
                print('\nexisting data has at least NSmin=%d samples per task, GPTune will generate at most NS-NSmin=%d additional samples.\n'%(NSmin,NS-NSmin))


        t3 = time.time_ns()

        t1 = time.time_ns()

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Multi-task Learning Autotuning """

        if (NS1 is not None and NS1>NS):
            raise Exception("NS1>NS")
        if (NS1 is None):
            NS1 = min(NS - 1, 3 * self.problem.DP) # heuristic rule in literature

        if(Igiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Igiven

########## normalize the data as the user always work in the original space
        if self.data.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
            tmp=[]
            for x in self.data.P:
                if(len(x)>0):
                    xNorm = self.problem.PS.transform(x)
                    tmp.append(xNorm)
                else:
                    tmp.append(np.empty( shape=(0, self.problem.DP) ))
            self.data.P=tmp
        if self.data.I is not None: # from a list of lists to a 2D numpy array
            self.data.I = self.problem.IS.transform(self.data.I)

        if (self.data.O is None and self.data.P is not None and self.data.I is not None): # tuning parameters and task parameters are given, but the output is none
            self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, self.data.P, self.data.D, self.historydb, options = kwargs)

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

        if (NSmin<NS1):
            check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
            tmpP = sampler.sample_parameters(n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    NSi = self.data.P[i].shape[0]
                    tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data

        if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
            raise Exception("len(self.data.O) !=len(self.data.I)")

        t2 = time.time_ns()
        time_sample_init = time_sample_init + (t2-t1)/1e9

        t1 = time.time_ns()
        if (NSmin<NS1):
            tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs)
            if(self.data.P is None): # no existing tuning data is available
                self.data.O = tmpO
                self.data.P = tmpP
            else:                
                for i in range(len(self.data.P)):
                    self.data.P[i] = np.vstack((self.data.P[i],tmpP[i]))
                    self.data.O[i] = np.vstack((self.data.O[i],tmpO[i]))

        t2 = time.time_ns()
        stats["func_eval_time"].append((t2-t1)/1e9)
        time_fun = time_fun + (t2-t1)/1e9

        modelers  = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
        searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options)')
        optiter = 0
        NSmin = min(map(len, self.data.P))
        while NSmin<NS:# YL: each iteration adds 1 (if single objective) or at most kwargs["search_more_samples"] (if multi-objective) sample until total #sample reaches NS

            if(self.problem.models_update is not None):
                ########## denormalize the data as the user always work in the original space
                tmpdata = copy.deepcopy(self.data)
                if tmpdata.I is not None:    # from 2D numpy array to a list of lists
                    tmpdata.I = self.problem.IS.inverse_transform(tmpdata.I)
                if tmpdata.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
                    tmpdata.P = [self.problem.PS.inverse_transform(x) for x in tmpdata.P]
                self.problem.models_update(tmpdata)
                self.data.D = tmpdata.D

            newdata = Data(problem = self.problem, I = self.data.I, D = self.data.D)
            print("MLA iteration: ",optiter)
            stats["modeling_iteration"].append(0)
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
                            if(self.problem.constants is not None):
                                points.update(self.problem.constants)
                            modeldata.append(self.problem.models(points))
                        modeldata=np.array(modeldata)
                        tmpdata.P[i] = np.hstack((tmpdata.P[i],modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
                for i in range(len(tmpdata.P)):   # LCM requires the same number of samples per task, so use the first NSmin samples
                    tmpdata.O[i] = tmpdata.O[i][0:NSmin,:]
                    tmpdata.P[i] = tmpdata.P[i][0:NSmin,:]
                # print(tmpdata.P[0])
                #print ("[bestxopt]: len: " + str(len(bestxopt)) + " val: " + str(bestxopt))
                if (kwargs["model_class"] == "Model_LCM"):
                    if (kwargs["model_output_constraint"] == True):
                        for i in range(len(tmpdata.O[0])):
                            out_of_range = False
                            for o_ in range(self.problem.DO):
                                output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                lower_bound = output_space["lower_bound"]
                                upper_bound = output_space["upper_bound"]
                                output_result = [copy.deepcopy(self.data.O[i][:,o_].reshape((-1,1))) for i in range(len(self.data.I))][0][i]
                                if output_result < lower_bound or \
                                   output_result > upper_bound:
                                    out_of_range = True

                            if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                tmpdata.O[0][i][0] = 1000000000.0 #sys.float_info.max

                    (bestxopt, neg_log_marginal_likelihood,
                            gradients, iteration) = \
                        modelers[o].train(data = tmpdata, **kwargs)
                    self.historydb.store_model_LCM(
                            o,
                            self.problem,
                            self.data.I,
                            bestxopt,
                            neg_log_marginal_likelihood,
                            gradients,
                            iteration)
                    stats["modeling_iteration"][optiter-1] += iteration
                else:
                    modelers[o].train(data = tmpdata, **kwargs)
                
                if self.options['verbose'] == True and self.options['model_class'] == 'Model_LCM' and len(self.data.I)>1:
                    C = modelers[o].M.kern.get_correlation_metric()
                    print("The correlation matrix C is \n", C)
                elif self.options['verbose'] == True and self.options['model_class'] == 'Model_GPy_LCM' and len(self.data.I)>1:
                    C = modelers[o].get_correlation_metric(len(self.data.I))
                    print("The correlation matrix C is \n", C)

            t2 = time.time_ns()
            stats["modeling_time"].append((t2-t1)/1e9)
            time_model = time_model + (t2-t1)/1e9

            t1 = time.time_ns()
            res = searcher.search_multitask(data = self.data, models = modelers, **kwargs)

            newdata.P = [x[1][0] for x in res]
            for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
                NSi = self.data.P[i].shape[0]
                newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]
            # print(more_samples,newdata.P)
            t2 = time.time_ns()
            stats["search_time"].append((t2-t1)/1e9)
            time_search = time_search + (t2-t1)/1e9

            t1 = time.time_ns()
            newdata.O = self.computer.evaluate_objective(problem = self.problem,
                    I = newdata.I,
                    P = newdata.P,
                    D = newdata.D,
                    history_db = self.historydb,
                    options = kwargs)
            t2 = time.time_ns()
            time_fun = time_fun + (t2-t1)/1e9
            self.data.merge(newdata)
            NSmin = min(map(len, self.data.P))

        # denormalize the data as the user always work in the original space
        if self.data.I is not None:    # from 2D numpy array to a list of lists
            self.data.I = self.problem.IS.inverse_transform(self.data.I)
        if self.data.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
            self.data.P = [self.problem.PS.inverse_transform(x) for x in self.data.P]

        t4 = time.time_ns()
        stats['time_total'] = (t4-t3)/1e9
        stats['time_fun'] = time_fun
        stats['time_model'] = time_model
        stats['time_search'] = time_search
        stats['time_sample_init'] = time_sample_init

        return (copy.deepcopy(self.data), modelers, stats)

    def MLA(self, NS, NS1 = None, NI = None, Igiven = None, **kwargs):
        if self.historydb.history_db is True:
            if self.historydb.load_func_eval == True and self.historydb.load_surrogate_model == True:
                return self.MLA_LoadModel(NS = NS, Igiven = Igiven)
            else:
                return self.MLA_HistoryDB(NS, NS1, NI, Igiven)

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
        O = self.computer.evaluate_objective(problem = self.problem, I = InewNorms, P =aprxoptsNormList, history_db = self.historydb, options = kwargs)
        t2 = time.time_ns()
        time_fun = time_fun + (t2-t1)/1e9

        #        print(aprxopts)
        #        pickle.dump(aprxopts, open('TLA1.pkl', 'w'))

        t4 = time.time_ns()
        stats['time_total'] = (t4-t3)/1e9
        stats['time_fun'] = time_fun

        return (aprxopts, O, stats)

    def TLA2(): # co-Kriging

        pass


class GPTune_MB(object):

    def __init__(self, tp : TuningProblem, computer : Computer = None, options : Options = None, **kwargs):

        """
        tp: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
        computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
        options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
        """

        # options contains: bin, bmax, eta
        # Or contains: fidelity map (dictionary from s values to budget level B(s)), eta
        if options['fidelity_map'] == None:
            self.smax = int(np.floor(np.log10(
                options['budget_max']/options['budget_min'])/np.log10(options['budget_base'])))
            self.budgets = [options['budget_max'] /
                            options['budget_base']**x for x in range(self.smax+1)]
            print(f'Using default budgets, smax = {self.smax}, budgets = {self.budgets}.')
        else:
            s_vals = sorted(options['fidelity_map'].keys())
            self.smax = s_vals[-1]
            self.budgets = [options['fidelity_map'].get(key) for key in s_vals]
            print(f'Using user-provided budgets, smax = {self.smax}, budgets = {self.budgets}.')

        parameter_space = tp.parameter_space
        output_space = tp.output_space
        objectives = tp.objective
        constraints = tp.constraints
        constants = tp.constants

        """ insert "budget" as the first dimension of the input space """
        inputs = [Real(options['budget_min']-1e-12,
                       options['budget_max'], transform="normalize", name="budget")]

        for n,p in zip(tp.input_space.dimension_names,tp.input_space.dimensions):
            if (isinstance(p, Real)):
                inputs.append(Real(p.bounds[0], p.bounds[1], transform="normalize", name=n))
            elif (isinstance(p, Integer)):
                inputs.append(Integer(p.bounds[0], p.bounds[1], transform="normalize", name=n))
            elif (isinstance(p, Categorical)):
                inputs.append(Categoricalnorm (list(p.bounds), transform="onehot", name=n))
            else:
                raise Exception("Unknown parameter type")

        print('inputs = ', inputs)
        input_space = Space(inputs)

        self.tp = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, None, constants=constants)
        self.computer = computer
        self.options  = options
        self.data     = Data(tp)


    def MB_LCM(self, NS=None, Igiven=None, Pdefault=None, **kwargs):
        """
        Igiven		 : a list of tasks 
        NS			 : number of samples in the highest budget arm
        Pdefault     : assuming there is a default parameter configuration among all tasks
        """

        np.set_printoptions(suppress=False, precision=4)
        print('\n\n\n------Starting MB_LCM (multi-arm bandit with LCM) with %d loops for task' % (NS), Igiven)

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

        # self.NSs = [int(self.options['budget_max']/x*NS) for x in self.budgets] # so that the highest fidelity has NS samples
        self.NSs = [int((self.smax+1)/(s+1))*self.options['budget_base']**s for s in range(self.smax+1)] # consistent with hyperband
        NSs1 = [0] * len(self.NSs)
        info = [[x, y] for x, y in zip(self.budgets, self.NSs)]
        print('total samples:', info)
                
        self.data.I = Igiven
        # self.data.D = [] 
        self.data.P = []
        self.data.O = []
        
        data = Data(self.tp)   # data contains all data in the extended task space returned by MLA

        for Nloop in range(NS):
            data1 = Data(self.tp)  # for each loop, data1 will have all data for all arms sampled by MLA (excluding SH samples)
            data1.I = []
            data1.P = []
            data1.O = []
            data1.D = []
            for s in range(len(self.budgets)):  # loop over the budget levels
                budget = self.budgets[s]
                ns = self.NSs[s] 
                ntotal = NSs1[s] + int(ns)
                NSs1[s] = NSs1[s] + ns
                print(f"Bracket s = {s}, budget = {budget}, ns = {ns}")
                newtasks = []
                if(self.data.D is not None):
                    data.D = []
                for s1 in range(s, len(self.budgets)):
                    for t in range(len(Igiven)):
                        budget1 = self.budgets[s1]
                        tmp = [budget1]+Igiven[t]
                        newtasks.append(tmp)
                        if(self.data.D is not None):
                            if(len(self.data.D)>0):
                                data.D.append(self.data.D[t])
                if s == 0 and Nloop == 0:
                    all_subtasks = copy.deepcopy(newtasks)

                data.I = newtasks
                if(Pdefault is not None):
                    data.P = [[Pdefault]] * len(newtasks) # as gt.MLA will load available database, Pdefault is effective only if database is empty   
                
                # print("Calling MLA: \ndata.I", data.I, "\ndata.P", data.P, "\ndata.O", data.O)
                # print(f"NS={ntotal}, Igiven={newtasks}, NI={len(newtasks)}, NS1={min(self.NSs)}")
                gt = GPTune(self.tp, computer=self.computer,
                            data=data, options=self.options)
                (data, _, stats0) = gt.MLA(NS=ntotal, Igiven=newtasks, NI=len(newtasks), NS1=min(self.NSs))
                data.P = [x[NSs1[s]-ns:NSs1[s]] for x in data.P]   # data has all data (including those from database), only takes ns ones newly generated by MLA 
                data.O = [x[NSs1[s]-ns:NSs1[s]] for x in data.O]
                data1.I += data.I[0:len(Igiven)]     # data1 has data generated by MLA, but only in the current arm
                data1.P += data.P[0:len(Igiven)]
                data1.O += data.O[0:len(Igiven)]
                if(data.D is not None):
                    data1.D += data.D[0:len(Igiven)]
                data.I=[]
                data.P=[]
                data.O=[]
                if(data.D is not None):
                    data.D=None
                # merge new results to history
                
                stats['time_total'] += stats0['time_total']
                stats['time_fun'] += stats0['time_fun']
                stats['time_model'] += stats0['time_model']
                stats['time_search'] += stats0['time_search']
                stats['time_sample_init'] += stats0['time_sample_init']
                # print(f'At the end of bracket {s}')
                # print('Current data1:')
                # print('data1.I: ', data1.I)
                # print('data1.P: ', data1.P)
                # print('data1.O: ', data1.O)
                # print("data1.D = ", data1.D)
                
            if Nloop == 0:
                self.data.P = data1.P[0:len(Igiven)]  # the first 0:len(Igiven) tasks of data1 are highest budget
                self.data.O = data1.O[0:len(Igiven)] 
            else:
                # self.data.P = [np.concatenate((self.data.P[i], data1.P[0:len(Igiven)][i])) for i in range(len(self.data.P))]
                self.data.P = [self.data.P[i] + data1.P[i] for i in range(len(self.data.P))]
                self.data.O = [np.concatenate((self.data.O[i], data1.O[i])) for i in range(len(self.data.O))]
            
            print("Finish multi-arm initial evaluation")
            # print('data.I: ', data.I)
            # print('data.P: ', data.P)
            # print('data.O: ', data.O)
            # print("data.D = ", data.D)
            print('self.data.P = ', self.data.P)
            print('self.data.O = ', self.data.O)
            
            print('\n\n\n------Start SH run on each arm, except the first highest fidelity arm')
            options1 = copy.deepcopy(self.options)
            kwargs.update(options1)
            for s in range(1, len(self.budgets)):
                print('\nArm s=%d'%(s))
                budget = self.budgets[s]
                ns = self.NSs[s]
                # print(f'\n\n\nArm {s}, Initial budget = {budget}, number of total samples = {ns}')
                ratio = int(ns/self.options['budget_base'])
                # print(f'Current s = {s}, budget = {budget}, ns = {ns}')
                idx = s*len(Igiven) 
                temp_I = data1.I[idx:idx+len(Igiven)]
                # print(f'Tasks: ', temp_I)
                temp_O = list(map(np.squeeze, data1.O[idx:idx+len(Igiven)]))
                # temp_P = list(map(np.array, data1.P[idx:idx+len(Igiven)]))
                temp_P = data1.P[idx:idx+len(Igiven)]
                for ri in range(s):
                    idx_sort = list(map(np.argsort, temp_O))
                    temp_O = [y[x[:ratio]] for (x, y) in zip(idx_sort, temp_O)]
                    temp_P = [[y[_i] for _i in x[:ratio]] for (x, y) in zip(idx_sort, temp_P)]
                    # temp_P = [x.tolist() for x in temp_P]
                    # budget *= self.options['budget_base'] # lift budget level
                    budget = self.budgets[s-ri-1]
                    for subtasks in temp_I:
                        subtasks[0] = budget
                    newdata = Data(problem=self.tp, I=temp_I, P=temp_P)
                    gt = GPTune(self.tp, computer=self.computer, data=newdata, options=self.options)
                    
                    
                    gt.history_db.load_history_func_eval(newdata, gt.problem, temp_I)


                    t1 = time.time_ns()
                    done=0
                    if(gt.data.P is not None):
                        if(len(gt.data.P[0])>=NSs1[s-ri-1]+ratio): # the evaluations have been done before 
                            done=1
                    if(done==0):
                        print(f'Evaluating top {ratio} by MLA with budget = {budget}')
                        newdata = Data(problem=self.tp, I=temp_I, P=temp_P)
                        gt = GPTune(self.tp, computer=self.computer, data=newdata, options=self.options)
                        tmp=[]
                        for x in gt.data.P:
                            if(len(x)>0):
                                xNorm = gt.problem.PS.transform(x)
                                tmp.append(xNorm)
                            else:
                                tmp.append(np.empty( shape=(0, gt.problem.DP) ))
                        gt.data.P=tmp
                        # print('gaga',gt.data.P[0])
                        gt.data.I = gt.problem.IS.transform(gt.data.I)
                        newdata.O = gt.computer.evaluate_objective(gt.problem, gt.data.I, gt.data.P, gt.data.D, gt.history_db, options = kwargs)
                        newdata.P = [gt.problem.PS.inverse_transform(x) for x in newdata.P]

                    else:
                        # print('done: what do you do')
                        newdata.I=gt.data.I
                        newdata.P=gt.data.P
                        newdata.O=gt.data.O
                        newdata.P = [elem[NSs1[s-ri-1]:NSs1[s-ri-1]+ratio] for elem in newdata.P]
                        newdata.O = [elem[NSs1[s-ri-1]:NSs1[s-ri-1]+ratio,:] for elem in newdata.O]

                    t2 = time.time_ns()
                    timefun=(t2-t1)/1e9
                    NSs1[s-ri-1] = NSs1[s-ri-1] + ratio
                    
                    temp_O = list(map(np.squeeze, newdata.O))
                    # temp_P = list(map(np.array, newdata.P))
                    temp_P = newdata.P
                    ratio = int(ratio/self.options['budget_base'])
                    stats['time_fun'] += timefun
                    stats['time_total'] += timefun
                # print(newdata.I,'jia')
                # print(newdata.P,'dia')
                newdata.I = Igiven
                # print('newdata before merge')
                # print('newdata.P = ', newdata.P)
                # print('self.data.P = ', self.data.P)
                # self.data.merge(newdata) # this would make self.data.P a list of numpy array
                self.data.P = [x + y for x, y in zip(self.data.P, newdata.P)]
                self.data.O = [np.concatenate((self.data.O[i], newdata.O[i])) for i in range(len(self.data.O))]
                
                # print('Data updated: ')
                # print('self.data.P = ', self.data.P)
                # print('self.data.O = ', self.data.O)
            
            print('Updated self.data after all SH runs')
            print('self.data.P = ', self.data.P)
            print('self.data.O = ', self.data.O)
            
            # data1_hist collects data1 from all loops (it excludes SH samples since it's used to locate arm and sample index of the optimal sample)
            data1.I = all_subtasks # change budgets back to initial values
            if Nloop == 0:
                data1_hist = copy.deepcopy(data1)             
            else:
                data1_hist.merge(data1)
                data1_hist.P = [elem.tolist() for elem in data1_hist.P]
                
            Nloop += 1
            print(f"Finish one loop, next Nloop = {Nloop}")
            print('data1_hist.I = ', data1_hist.I)    
            print('data1_hist.P = ', data1_hist.P)
            print('data1_hist.O = ', data1_hist.O)
            
        return (copy.deepcopy(self.data), stats, data1_hist)

#### Wrapper Functions

def LoadSurrogateModelData(meta_path=None, meta_dict=None, tuning_configuration:dict = None):

    meta_data = {}

    if meta_path != None:
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f_in:
                    meta_data.update(json.load(f_in))
            except:
                print ("[Error] not able to get model load configuration from path")

    if meta_dict != None:
        try:
            meta_data.update(meta_dict)
        except:
            print ("[Error] not able to get model load configuration from dict")

    input_space_given = meta_data["input_space"]
    parameter_space_given = meta_data["parameter_space"]
    output_space_given = meta_data["output_space"]
    task_parameters_given = meta_data["task_parameters"]
    print ("task_parameters_given: ", task_parameters_given)

    if "modeler" in meta_data:
        modeler = meta_data['modeler']
    else:
        modeler = 'Model_LCM'

    historydb = HistoryDB()
    model_data = historydb.load_surrogate_model_meta_data(
            task_parameters_given,
            tuning_configuration,
            input_space_given,
            parameter_space_given,
            output_space_given,
            0,
            modeler)

    print ("MODEL DATA: ", model_data)

    return (model_data)

def CreateGPTuneFromModelData(model_data):

    input_space_info = model_data["input_space"]
    parameter_space_info = model_data["parameter_space"]
    output_space_info = model_data["output_space"]

    input_space_arr = []
    for input_space_info in model_data["input_space"]:
        name_ = input_space_info["name"]
        type_ = input_space_info["type"]
        transformer_ = input_space_info["transformer"]

        if type_ == "int" or type_ == "Int" or type_ == "Integer" or type_ == "integer":
            lower_bound_ = input_space_info["lower_bound"]
            upper_bound_ = input_space_info["upper_bound"]
            input_space = Integer(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            input_space_arr.append(input_space)
        elif type_ == "real" or type_ == "Real" or type_ == "float" or type_ == "Float":
            lower_bound_ = input_space_info["lower_bound"]
            upper_bound_ = input_space_info["upper_bound"]
            input_space = Real(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            input_space_arr.append(input_space)
        elif type_ == "categorical" or type_ == "Categorical" or type_ == "category" or type_ == "Category":
            categories = input_space_info["categories"]
            input_space = Categoricalnorm(categories, transform=transformer_, name=name_)
            input_space_arr.append(input_space)
    IS = Space(input_space_arr)

    parameter_space_arr = []
    for parameter_space_info in model_data["parameter_space"]:
        name_ = parameter_space_info["name"]
        type_ = parameter_space_info["type"]
        transformer_ = parameter_space_info["transformer"]

        if type_ == "int" or type_ == "Int" or type_ == "Integer" or type_ == "integer":
            lower_bound_ = parameter_space_info["lower_bound"]
            upper_bound_ = parameter_space_info["upper_bound"]
            parameter_space = Integer(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            parameter_space_arr.append(parameter_space)
        elif type_ == "real" or type_ == "Real" or type_ == "float" or type_ == "Float":
            lower_bound_ = parameter_space_info["lower_bound"]
            upper_bound_ = parameter_space_info["upper_bound"]
            parameter_space = Real(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            parameter_space_arr.append(parameter_space)
        elif type_ == "categorical" or type_ == "Categorical" or type_ == "category" or type_ == "Category":
            categories = parameter_space_info["categories"]
            parameter_space = Categoricalnorm(categories, transform=transformer_, name=name_)
            parameter_space_arr.append(parameter_space)
    PS = Space(parameter_space_arr)

    output_space_arr = []
    for output_space_info in model_data["output_space"]:
        name_ = output_space_info["name"]
        type_ = output_space_info["type"]
        transformer_ = output_space_info["transformer"]

        if type_ == "int" or type_ == "Int" or type_ == "Integer" or type_ == "integer":
            lower_bound_ = output_space_info["lower_bound"]
            upper_bound_ = output_space_info["upper_bound"]
            output_space = Integer(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            output_space_arr.append(output_space)
        elif type_ == "real" or type_ == "Real" or type_ == "float" or type_ == "Float":
            lower_bound_ = output_space_info["lower_bound"]
            upper_bound_ = output_space_info["upper_bound"]
            output_space = Real(lower_bound_, upper_bound_, transform=transformer_, name=name_)
            output_space_arr.append(output_space)
        elif type_ == "categorical" or type_ == "Categorical" or type_ == "category" or type_ == "Category":
            categories = output_space_info["categories"]
            output_space = Category(categories, transform=transformer_, name=name_)
            output_space_arr.append(output_space)
    OS = Space(output_space_arr)

    problem = TuningProblem(IS, PS, OS, objective=None, constraints=None, models=None, constants=None)
    computer = Computer(nodes=1, cores=2) # number of nodes/cores is not actually used when reproducing only surrogate models
    data = Data(problem)
    options = Options()
    if "modeler" in model_data:
        options['model_class'] = model_data['modeler']
    else:
        options['model_class'] = 'Model_LCM'
    gt = GPTune(problem, computer=computer, data=data, options=options)

    return (gt)

def LoadSurrogateModelFunction(meta_path="./.gptune/model.json", meta_dict=None, tuning_configuration:dict = None):

    model_data = LoadSurrogateModelData(meta_path, meta_dict, tuning_configuration)
    gt = CreateGPTuneFromModelData(model_data)
    (models, model_function) = gt.LoadSurrogateModel(model_data = model_data)

    return (model_function)

def SensitivityAnalysis(model_data:dict=None, task_parameters=None, num_samples:int=1000):

    gt = CreateGPTuneFromModelData(model_data)
    (models, model_function) = gt.LoadSurrogateModel(model_data = model_data)

    from SALib.sample import saltelli
    from SALib.analyze import sobol

    num_vars = len(model_data["parameter_space"])
    parameter_names = []
    parameter_bounds = []
    parameter_types = []
    categorical_parameter_values = {}
    for parameter_info in model_data["parameter_space"]:
        parameter_name = parameter_info["name"]
        parameter_type = parameter_info["type"]

        parameter_names.append(parameter_name)
        parameter_types.append(parameter_type)

        if parameter_type == "int" or parameter_type == "real":
            lower_bound = parameter_info["lower_bound"]
            upper_bound = parameter_info["upper_bound"]
            parameter_bounds.append([lower_bound, upper_bound])
        elif parameter_type == "categorical":
            lower_bound = 0
            upper_bound = len(parameter_info["categories"])
            parameter_bounds.append([lower_bound, upper_bound])
            categorical_parameter_values[parameter_name] = parameter_info["categories"]
    print ("Categorical_parameter_values: ", categorical_parameter_values)

    parameter_categories = []
    problem = {
            'num_vars': num_vars,
            'names': parameter_names,
            'bounds': parameter_bounds,
            }

    # Generate new samples for sensitivity analysis from the fitted surrogate model.
    parameter_values = saltelli.sample(problem, num_samples)

    Y = []
    for i in range(len(parameter_values)):
        model_input = {}
        for j in range(num_vars):
            if parameter_types[j] == "int" or parameter_types[j] == "real":
                model_input[parameter_names[j]] = int(parameter_values[i][j])
            elif parameter_types[j] == "categorical":
                model_input[parameter_names[j]] = categorical_parameter_values[parameter_names[j]][int(parameter_values[i][j])]

        num_task_parameters = len(model_data["input_space"])
        for j in range(num_task_parameters):
            model_input[model_data["input_space"][j]["name"]] = task_parameters[j]
        #print (model_input)

        output_name = model_data["output_space"][0]["name"]

        Y.append(model_function(model_input)[output_name][0][0])

    print (Y)

    # Sensitivity analysis based on the samples drawn from the fitted surrogate model.
    Si = sobol.analyze(problem, np.array(Y), print_to_console=True)
    #print (Si)
    return Si

def GetSurrogateModelConfigurations(meta_path="./.gptune/model.json", meta_dict=None):

    meta_data = {}

    if meta_path != None:
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f_in:
                    meta_data.update(json.load(f_in))
            except:
                print ("[Error] not able to get model load configuration from path")

    if meta_dict != None:
        try:
            meta_data.update(meta_dict)
        except:
            print ("[Error] not able to get model load configuration from dict")

    loadable_machine_configurations = meta_data["loadable_machine_configurations"]
    loadable_software_configurations = meta_data["loadable_software_configurations"]

    input_space_given = meta_data["input_space"]
    parameter_space_given = meta_data["parameter_space"]
    output_space_given = meta_data["output_space"]
    task_parameters_given = meta_data["task_parameters"]

    if "modeler" in meta_data:
        modeler = meta_data['modeler']
    else:
        modeler = 'Model_LCM'

    historydb = HistoryDB()
    (model_configurations) = historydb.load_surrogate_model_configurations(
            task_parameters_given,
            input_space_given,
            parameter_space_given,
            output_space_given,
            loadable_machine_configurations,
            loadable_software_configurations,
            0,
            modeler)

    return model_configurations

