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

import numpy as np

import json
from filelock import Timeout, FileLock
from operator import mul
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
        self.models_transfer = None

    #def GenSurrogateModel(self, model_data : dict, function_evaluations : dict, **kwargs):
    def GenSurrogateModel(self, task_parameters, function_evaluations, **kwargs):

        kwargs.update(self.options)

        Tgiven = task_parameters

        """ Load history function evaluation data """

        if len(function_evaluations) == 0:
            print ("no history data has been loaded")
            return

        num_tasks = len(Tgiven)

        PS_history = [[] for i in range(num_tasks)]
        OS_history = [[] for i in range(num_tasks)]
        for func_eval in function_evaluations:
            parameter_arr = []
            for k in range(len(self.problem.PS)):
                if type(self.problem.PS[k]).__name__ == "Categoricalnorm":
                    parameter_arr.append(str(func_eval["tuning_parameter"][self.problem.PS[k].name]))
                elif type(self.problem.PS[k]).__name__ == "Integer":
                    parameter_arr.append(int(func_eval["tuning_parameter"][self.problem.PS[k].name]))
                elif type(self.problem.PS[k]).__name__ == "Real":
                    parameter_arr.append(float(func_eval["tuning_parameter"][self.problem.PS[k].name]))
                else:
                    parameter_arr.append(func_eval["tuning_parameter"][self.problem.PS[k].name])
            task_id = self.historydb.search_func_eval_task_id(func_eval, self.problem, Tgiven)
            if task_id >= 0:
                PS_history[task_id].append(parameter_arr)
                OS_history[task_id].append(\
                    [func_eval["evaluation_result"][self.problem.OS[k].name] \
                    for k in range(len(self.problem.OS))])
        self.data.I = Tgiven
        self.data.P = PS_history
        self.data.O=[]
        for i in range(len(OS_history)):
            if(len(OS_history[i])==0):
                self.data.O.append(np.empty( shape=(0, self.problem.DO)))
            else:
                self.data.O.append(np.array(OS_history[i]))
                if(any(ele==[None] for ele in OS_history[i])):
                    print ("history data contains null function values")
                    exit()

        """ Update data space """

        if(Tgiven is not None and self.data.I is None):
            self.data.I = Tgiven

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
        tmpdata = copy.deepcopy(self.data)
        for o in range(self.problem.DO):
            modelers[o].train(data = tmpdata, **kwargs)

        #for i in range(self.problem.DO):
        #    modelers[i].gen_model_from_hyperparameters(self.data,
        #            model_data["hyperparameters"],
        #            **kwargs)

        def model_function(point):

            task_parameter_names = [self.problem.IS[k].name for k in range(len(self.problem.IS))]
            tuning_parameter_names = [self.problem.PS[k].name for k in range(len(self.problem.PS))]
            tuning_parameter_types = [type(self.problem.PS[k]).__name__ for k in range(len(self.problem.PS))]
            output_names = [self.problem.OS[k].name for k in range(len(self.problem.OS))]

            tid = 0

            #print ("task_parameter_names: ", task_parameter_names)
            #print ("tuning_parameter_names: ", tuning_parameter_names)
            #print ("tuning_parameter_types: ", tuning_parameter_types)
            #print ("output_names: ", output_names)
            #print ("Tgiven: ", Tgiven)
            #print ("point: ", point)

            input_task = Tgiven[tid]

            #input_task = []
            #for task_parameter_name in task_parameter_names:
            #    input_task.append(point[task_parameter_name])

            #tid = -1
            #for i in range(len(Tgiven)):
            #    is_equal = True
            #    for j in range(len(Tgiven[i])):
            #        if Tgiven[i][j] != input_task[j]:
            #            is_equal = False
            #    if (is_equal):
            #        tid = i
            #        break
            #if tid == -1:
            #    print ("[Error] cannot find model for the given input task: ", input_task)
            #    return None

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

    def LoadSurrogateModel(self, model_data : dict, **kwargs):

        Tgiven = model_data["task_parameters"]

        """ Load history function evaluation data """

        self.historydb.load_model_func_eval(data = self.data, problem = self.problem,
                Tgiven = Tgiven, model_data = model_data)

        """ Update data space """

        if(Tgiven is not None and self.data.I is None):
            self.data.I = Tgiven

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
            if kwargs["model_class"] == 'Model_LCM':
                modelers[i].gen_model_from_hyperparameters(self.data,
                        model_data["hyperparameters"],
                        **kwargs)
            elif kwargs["model_class"] == "Model_GPy_LCM":
                modelers[i].gen_model_from_hyperparameters(self.data,
                        model_data["hyperparameters"],
                        model_data["modeling_options"],
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
            #print ("Tgiven: ", Tgiven)

            input_task = []
            for task_parameter_name in task_parameter_names:
                input_task.append(point[task_parameter_name])

            tid = -1
            for i in range(len(Tgiven)):
                is_equal = True
                for j in range(len(Tgiven[i])):
                    if Tgiven[i][j] != input_task[j]:
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

    def MLA_LoadModel(self, NS = 0, Tgiven = None, method = "max_evals", update = 0, model_uid = None, **kwargs):
        print('\n\n\n------Starting MLA with Trained Model for %d tasks and %d samples each '%(len(Tgiven),NS))
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
        self.historydb.load_history_func_eval(self.data, self.problem, Tgiven)
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
        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

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
            if model_uid == None:
                #TODO CHECK: make self.data is correct (we may need to load (or double check) func eval data based on the model data)

                if method == "max_evals":
                    (hyperparameters, parameter_names, model_options) = self.historydb.load_max_evals_surrogate_model_hyperparameters(
                            self.tuningproblem, Tgiven, i, kwargs["model_class"])
                elif method == "MLE" or method == "mle":
                    (hyperparameters, parameter_names, model_options) = self.historydb.load_MLE_surrogate_model_hyperparameters(
                            self.tuningproblem, Tgiven, i, kwargs["model_class"])
                elif method == "AIC" or method == "aic":
                    (hyperparameters, parameter_names, model_options) = self.historydb.load_AIC_surrogate_model_hyperparameters(
                            self.tuningproblem, Tgiven, i, kwargs["model_class"])
                elif method == "BIC" or method == "bic":
                    (hyperparameters, parameter_names, model_options) = self.historydb.load_BIC_model_hyperparameters(
                            self.tuningproblem, Tgiven, i, kwargs["model_class"])
                else:
                    (hyperparameters, parameter_names, model_options) = self.historydb.load_max_evals_surrogate_model_hyperparameters(
                            self.tuningproblem, Tgiven, i, kwargs["model_class"])

            else:
                (hyperparameters, parameter_names, model_options) = self.historydb.load_surrogate_model_hyperparameters_by_uid(model_uid)

            modelers[i].gen_model_from_hyperparameters(self.data,
                    hyperparameters,
                    model_options,
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
            print("Iteration: ",optiter)
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
                        (hyperparameters, modeling_options, model_stats) = modelers[o].train(data = tmpdata, **kwargs)
                        self.historydb.store_model_GPy_LCM(
                                o,
                                self.problem,
                                self.data.I,
                                hyperparameters,
                                modeling_options,
                                model_stats)
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

    # T: task parameter (1-d np array)
    # P: given tuning parameters (manual input) to evaluate (2-d np array)
    def EvaluateObjective(self, T: np.ndarray = None, P: np.ndarray = None, **kwargs):

        options = copy.deepcopy(self.options)
        kwargs.update(options)

        Tgiven = self.problem.IS.transform([T])[0]

        for i in range(len(P)):
            xNorm = self.problem.PS.transform([P[i]])
            self.computer.evaluate_objective_onetask(problem=self.problem, T2=Tgiven, P2=xNorm, history_db=self.historydb, options=kwargs)

        return

    def MLA_(self, NS, NS1 = None, NI = None, Tgiven = None, T_sampleflag = None, function_evaluations = None, source_function_evaluations = None, models_transfer = None, **kwargs):
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

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            # load function evaluations regardless of the modeling scheme of the sample
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven, function_evaluations, source_function_evaluations=source_function_evaluations, options=None)

            ## in case source function evaluation data is used for transfer learning
            #if source_function_evaluations != None:
            #    print ("DATA:P: ", self.data.P)
            #    self.historydb.load_source_function_evaluations(self.data, self.problem, Tgiven, num_target_task=NI-len(source_function_evaluations), source_function_evaluations=source_function_evaluations)

        if (NI is None and self.data.I is not None):
           NI = len(self.data.I)
        if (T_sampleflag is None):
           T_sampleflag = [ True for i in range (NI)]     
        T_bit_mask = list(map(int, T_sampleflag)) 
        tids=[]
        for i in range(NI):
            if(T_sampleflag[i] is True):
                tids.append(i)

        np.set_printoptions(suppress=False,precision=4)
        NSmin=0
        NSmax=0
        if (self.data.P is not None):
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active) # the number of samples per task in existing tuning data can be different
            NSmax = max(NS_active)

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

        """ Multi-task Learning Autotuning """

        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

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

        Ptmp = copy.deepcopy(self.data.P)
        if self.data.P is not None:
            for i in range(len(Ptmp)):
                if(T_sampleflag[i] is False):
                    Ptmp[i] = np.empty(shape=(0,self.problem.DP))

        if (self.data.O is None and Ptmp is not None and self.data.I is not None): # tuning parameters and task parameters are given, but the output is none
            self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, Ptmp, self.data.D, self.historydb, options = kwargs)

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

        print ("NS1: ", NS1)
        is_pilot = False
        run_pilot_anyway = False
        if NS1 == 0 and models_transfer != None:
            NS1 = 1
            option_tla = copy.deepcopy(self.options)
            option_tla["TLA_method"] = "Regression"
            searcher_tla = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = option_tla, models_transfer = models_transfer)')
            res = searcher_tla.search_multitask(data = self.data, models = None, **kwargs)
            tmpP = [x[1][0] for x in res]
            #print ("tmpP: ", tmpP)

            for i in range(NI):
                if(T_sampleflag[i] is False):
                    tmpP[i] = np.empty(shape=(0,self.problem.DP))

            run_pilot_anyway = False
            if (kwargs["model_input_separation"] == True or kwargs["model_peeking_level"] > 1):
                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is None:
                    # no samples from this modeling approach
                    run_pilot_anyway = True

            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    if(run_pilot_anyway == False and T_sampleflag[i] is True):
                        NSi = self.data.P[i].shape[0]
                        tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data
        else:
            is_pilot = True
            if NS1 == 0:
                NS1 = 1
            if (NSmin<NS1):
                check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
                tmpP = sampler.sample_parameters(problem = self.problem, n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
                for i in range(NI):
                    if(T_sampleflag[i] is False):
                        tmpP[i] = np.empty(shape=(0,self.problem.DP))
                # print ("tmpP: ", tmpP)

                if(self.data.P is not None):
                    for i in range(len(self.data.P)):
                        if(T_sampleflag[i] is True):
                            NSi = self.data.P[i].shape[0]
                            tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data

        if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
            raise Exception("len(self.data.O) !=len(self.data.I)")

        t2 = time.time_ns()
        time_sample_init = time_sample_init + (t2-t1)/1e9

        t1 = time.time_ns()

        if (NSmin<NS1 or run_pilot_anyway == True):
            tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs, is_pilot=is_pilot)
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
        if self.data.P != None:
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active)
        else:
            NSmin = 0
        print ("NSmin: ", NSmin)
        print ("NS: ", NS)
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
            print("Iteration: ",optiter)
            stats["modeling_iteration"].append(0)
            optiter = optiter + 1
            t1 = time.time_ns()
            for o in range(self.problem.DO):

                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
                    tmp=[]
                    for x in tmpdata.P:
                        if(len(x)>0):
                            xNorm = self.problem.PS.transform(x)
                            tmp.append(xNorm)
                        else:
                            tmp.append(np.empty( shape=(0, self.problem.DP) ))
                    tmpdata.P=tmp
                if(tmpdata.P is not None):
                    for i in range(len(tmpdata.P)):
                        if(T_sampleflag[i] is False and tmpdata.P[i].shape[0]==0):
                            tmpdata.P[i] = copy.deepcopy(self.data.P[i])
                            tmpdata.O[i] = copy.deepcopy(self.data.O[i])

                    if tmpdata.I is not None: # from a list of lists to a 2D numpy array
                        tmpdata.I = self.problem.IS.transform(tmpdata.I)
                else:
                    tmpdata = copy.deepcopy(self.data)
                    tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]

                #print ("tmpdata.I: ", tmpdata.I)
                #print ("self data P: ", self.data.P)
                #print ("tmpdata.P: ", tmpdata.P)
                #print ("tmpdata.O: ", tmpdata.O)
                #print ("tmpdata.P: ", tmpdata.P)

                if (kwargs["model_output_constraint"] != None):
                    tmp_tmpdata = Data(self.problem)
                    tmp_tmpdata.I = copy.deepcopy(self.data.I)
                    tmp_tmpdata.P = [[] for i in range(len(self.data.P))]
                    tmp_tmpdata.O = [[] for i in range(len(self.data.O))]
                    tmp_tmpdata.D = copy.deepcopy(self.data.D)

                    for t in range(len(tmpdata.O)):
                        for i in range(len(tmpdata.O[t])):
                            out_of_range = False
                            for o_ in range(self.problem.DO):
                                output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                lower_bound = output_space["lower_bound"]
                                upper_bound = output_space["upper_bound"]
                                output_result = [copy.deepcopy(self.data.O[i][:,o_].reshape((-1,1))) for i in range(len(self.data.I))][t][i]
                                if output_result < lower_bound or \
                                   output_result > upper_bound:
                                    out_of_range = True
                            if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                if (kwargs["model_output_constraint"] == 'LargeNum'):
                                    tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                    tmp_tmpdata.O[t].append([1000000000.0]) #sys.float_info.max
                                elif (kwargs["model_output_constraint"] == 'Ignore'):
                                    pass
                            else:
                                tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                tmp_tmpdata.O[t].append(tmpdata.O[t][i])

                    for t in range(len(tmpdata.O)):
                        tmp_tmpdata.P[t] = np.array(tmp_tmpdata.P[t])
                        tmp_tmpdata.O[t] = np.array(tmp_tmpdata.O[t])

                    for t in range(len(tmp_tmpdata.O)):
                        if len(tmp_tmpdata.O[t]) > 0:
                            tmpdata.P[t] = copy.deepcopy(tmp_tmpdata.P[t])
                            tmpdata.O[t] = copy.deepcopy(tmp_tmpdata.O[t])

                if(self.problem.models is not None):
                    for i in range(len(tmpdata.P)):
                        t = tmpdata.I[i]
                        I_orig = self.problem.IS.inverse_transform(np.array(t, ndmin=2))[0]
                        points1 = {self.problem.IS[k].name: I_orig[k] for k in range(self.problem.DI)}
                        modeldata=[]
                        for p in range(len(tmpdata.P[i])):
                            x = tmpdata.P[i][p]
                            x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                            points = {self.problem.PS[k].name: x_orig[k] for k in range(self.problem.DP)}
                            points.update(points1)
                            if(tmpdata.D is not None):
                                points.update(tmpdata.D[i])
                            if(self.problem.constants is not None):
                                points.update(self.problem.constants)
                            modeldata.append(self.problem.models(points))
                        modeldata=np.array(modeldata)
                        tmpdata.P[i] = np.hstack((tmpdata.P[i],modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space

                for i in range(len(tmpdata.P)):
                    if(T_sampleflag[i] is False and tmpdata.P[i].shape[0]==0):
                        raise Exception("T_sampleflag[%d] is False but self.data.P[%d] has no data"%(i,i))

                # print(tmpdata.P[0])
                #print ("[bestxopt]: len: " + str(len(bestxopt)) + " val: " + str(bestxopt))
                if (kwargs["model_class"] == "Model_LCM"):
                    # YC: [TODO-check] NSmin checking (with updated NSmin) still necessary? In case of options['model_output_constraint']='Ignore', the updated tmpdata.P and tmpdata.O can have unequal number of samples, but seems like it's running anyways. For now, we can say options['model_output_constraint']='Ignore' is not recommended for Model_LCM mode.
                    for i in range(len(tmpdata.P)):   # LCM requires the same number of samples per task, so use the first NSmin samples
                        tmpdata.O[i] = tmpdata.O[i][0:NSmin,:]
                        tmpdata.P[i] = tmpdata.P[i][0:NSmin,:]

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
                    (hyperparameters, modeling_options, model_stats) = modelers[o].train(data = tmpdata, **kwargs)
                    self.historydb.store_model_GPy_LCM(
                            o,
                            self.problem,
                            self.data.I,
                            hyperparameters,
                            modeling_options,
                            model_stats)

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
            res = searcher.search_multitask(data = self.data, models = modelers, tids=tids, **kwargs)
            newdata.P=[]
            i1=0
            for i in range(NI):
                if(T_sampleflag[i] is True):
                    newdata.P.append(res[i1][1][0])
                    i1=i1+1
                    NSi = self.data.P[i].shape[0]
                    newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:] # if NSi>=NS, skip the function evaluation                    
                else:
                    newdata.P.append(np.empty(shape=(0,self.problem.DP)))
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
            # print(self.data.P)
            # print(list(map(mul,T_bit_mask,list(map(len, self.data.P)))))
            # print(tids)
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active)

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

    def SLA(self, NS, NS1 = None, Tgiven = None):

        if NS1 == None:
            NS1 = int(NS/2)
        elif NS1 > NS:
            raise Exception("NS1>NS")

        print("\n\n\n------Starting SLA (%d samples)"%(NS))

        print("self.options: ", self.options)

        if self.options["BO_objective_evaluation_parallelism"] == False:
            (data, modeler, stats) = self.MLA_(NS, NS1, 1, [Tgiven], T_sampleflag=[True], function_evaluations=None, source_function_evaluations=None, models_transfer=None)
        elif self.options["BO_objective_evaluation_parallelism"] == True:
            (data, modeler, stats) = self.MLA_ParallelEval_(NS, NS1, 1, [Tgiven], T_sampleflag=[True], function_evaluations=None, source_function_evaluations=None, models_transfer=None)
        else:
            (data, modeler, stats) = self.MLA_(NS, NS1, 1, [Tgiven], T_sampleflag=[True], function_evaluations=None, source_function_evaluations=None, models_transfer=None)

        data.I = data.I[0]
        data.P = data.P[0]
        data.O = data.O[0]
        data.D = data.D[0]

        return data, modeler, stats

    def MLA_ParallelEval_(self, NS, NS1 = None, NI = None, Tgiven = None, T_sampleflag = None, function_evaluations = None, source_function_evaluations = None, models_transfer = None, **kwargs):
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

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            # load function evaluations regardless of the modeling scheme of the sample
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven, function_evaluations, source_function_evaluations=source_function_evaluations, options=None)

            ## in case source function evaluation data is used for transfer learning
            #if source_function_evaluations != None:
            #    print ("DATA:P: ", self.data.P)
            #    self.historydb.load_source_function_evaluations(self.data, self.problem, Tgiven, num_target_task=NI-len(source_function_evaluations), source_function_evaluations=source_function_evaluations)

        if (NI is None and self.data.I is not None):
           NI = len(self.data.I)
        if (T_sampleflag is None):
           T_sampleflag = [ True for i in range (NI)]
        T_bit_mask = list(map(int, T_sampleflag))
        tids=[]
        for i in range(NI):
            if(T_sampleflag[i] is True):
                tids.append(i)

        np.set_printoptions(suppress=False,precision=4)
        NSmin=0
        NSmax=0
        if (self.data.P is not None):
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active) # the number of samples per task in existing tuning data can be different
            NSmax = max(NS_active)

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

        """ Multi-task Learning Autotuning """

        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

        # normalize the data as the user always work in the original space
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

        Ptmp = copy.deepcopy(self.data.P)
        if self.data.P is not None:
            for i in range(len(Ptmp)):
                if(T_sampleflag[i] is False):
                    Ptmp[i] = np.empty(shape=(0,self.problem.DP))

        if (self.data.O is None and Ptmp is not None and self.data.I is not None): # tuning parameters and task parameters are given, but the output is none
            self.data.O = self.computer.evaluate_objective(self.problem, self.data.I, Ptmp, self.data.D, self.historydb, options = kwargs)

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

        print ("NS1: ", NS1)
        is_pilot = False
        run_pilot_anyway = False
        if NS1 == 0 and models_transfer != None:
            NS1 = 1
            option_tla = copy.deepcopy(self.options)
            option_tla["TLA_method"] = "Regression"
            searcher_tla = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = option_tla, models_transfer = models_transfer)')
            res = searcher_tla.search_multitask(data = self.data, models = None, **kwargs)
            tmpP = [x[1][0] for x in res]
            #print ("tmpP: ", tmpP)

            for i in range(NI):
                if(T_sampleflag[i] is False):
                    tmpP[i] = np.empty(shape=(0,self.problem.DP))

            run_pilot_anyway = False
            if (kwargs["model_input_separation"] == True or kwargs["model_peeking_level"] > 1):
                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is None:
                    # no samples from this modeling approach
                    run_pilot_anyway = True

            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    if(run_pilot_anyway == False and T_sampleflag[i] is True):
                        NSi = self.data.P[i].shape[0]
                        tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data
        else:
            is_pilot = True
            if NS1 == 0:
                NS1 = 1
            if (NSmin<NS1):
                check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
                tmpP = sampler.sample_parameters(problem = self.problem, n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
                for i in range(NI):
                    if(T_sampleflag[i] is False):
                        tmpP[i] = np.empty(shape=(0,self.problem.DP))
                # print ("tmpP: ", tmpP)

                if(self.data.P is not None):
                    for i in range(len(self.data.P)):
                        if(T_sampleflag[i] is True):
                            NSi = self.data.P[i].shape[0]
                            tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data

        if (self.data.O is not None and len(self.data.O) !=len(self.data.I)):
            raise Exception("len(self.data.O) !=len(self.data.I)")

        t2 = time.time_ns()
        time_sample_init = time_sample_init + (t2-t1)/1e9

        t1 = time.time_ns()

        if (NSmin<NS1 or run_pilot_anyway == True):
            tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs, is_pilot=is_pilot)
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
        if self.data.P != None:
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active)
        else:
            NSmin = 0

        num_evaluation_instances = 1
        if kwargs["distributed_memory_parallelism"] == True:
            num_evaluation_instances = kwargs["objective_multisample_processes"]
        elif kwargs["shared_memory_parallelism"] == True:
            num_evaluation_instances = kwargs["objective_multisample_threads"]

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
            newdata.P=[[] for i in range(NI)]

            data_replica = copy.deepcopy(self.data)

            if NS-NSmin < num_evaluation_instances:
                num_evaluation_instances = NS-NSmin

            for evaluation_instance in range(num_evaluation_instances):

                print("Iteration: ",optiter)
                stats["modeling_iteration"].append(0)

                optiter = optiter + 1
                t1 = time.time_ns()

                for o in range(self.problem.DO):
                    tmpdata = Data(self.problem)
                    if evaluation_instance == 0:
                        self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                        if tmpdata.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
                            tmp=[]
                            for x in tmpdata.P:
                                if(len(x)>0):
                                    xNorm = self.problem.PS.transform(x)
                                    tmp.append(xNorm)
                                else:
                                    tmp.append(np.empty( shape=(0, self.problem.DP) ))
                            tmpdata.P=tmp
                        if(tmpdata.P is not None):
                            for i in range(len(tmpdata.P)):
                                if(T_sampleflag[i] is False and tmpdata.P[i].shape[0]==0):
                                    tmpdata.P[i] = copy.deepcopy(data_replica.P[i])
                                    tmpdata.O[i] = copy.deepcopy(data_replica.O[i])

                            if tmpdata.I is not None: # from a list of lists to a 2D numpy array
                                tmpdata.I = self.problem.IS.transform(tmpdata.I)
                        else:
                            tmpdata = copy.deepcopy(data_replica)
                            tmpdata.O = [copy.deepcopy(data_replica.O[i][:,o].reshape((-1,1))) for i in range(len(data_replica.I))]
                    else:
                        tmpdata = copy.deepcopy(data_replica)
                        tmpdata.O = [copy.deepcopy(data_replica.O[i][:,o].reshape((-1,1))) for i in range(len(data_replica.I))]

                    if (kwargs["model_output_constraint"] != None):
                        tmp_tmpdata = Data(self.problem)
                        tmp_tmpdata.I = copy.deepcopy(data_replica.I)
                        tmp_tmpdata.P = [[] for i in range(len(data_replica.P))]
                        tmp_tmpdata.O = [[] for i in range(len(data_replica.O))]
                        tmp_tmpdata.D = copy.deepcopy(data_replica.D)

                        for t in range(len(tmpdata.O)):
                            for i in range(len(tmpdata.O[t])):
                                out_of_range = False
                                for o_ in range(self.problem.DO):
                                    output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                    lower_bound = output_space["lower_bound"]
                                    upper_bound = output_space["upper_bound"]
                                    output_result = [copy.deepcopy(data_replica.O[i][:,o_].reshape((-1,1))) for i in range(len(data_replica.I))][t][i]
                                    if output_result < lower_bound or \
                                       output_result > upper_bound:
                                        out_of_range = True
                                if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                    if (kwargs["model_output_constraint"] == 'LargeNum'):
                                        tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                        tmp_tmpdata.O[t].append([1000000000.0]) #sys.float_info.max
                                    elif (kwargs["model_output_constraint"] == 'Ignore'):
                                        pass
                                else:
                                    tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                    tmp_tmpdata.O[t].append(tmpdata.O[t][i])

                        for t in range(len(tmpdata.O)):
                            tmp_tmpdata.P[t] = np.array(tmp_tmpdata.P[t])
                            tmp_tmpdata.O[t] = np.array(tmp_tmpdata.O[t])

                        for t in range(len(tmp_tmpdata.O)):
                            if len(tmp_tmpdata.O[t]) > 0:
                                tmpdata.P[t] = copy.deepcopy(tmp_tmpdata.P[t])
                                tmpdata.O[t] = copy.deepcopy(tmp_tmpdata.O[t])

                    if(self.problem.models is not None):
                        for i in range(len(tmpdata.P)):
                            t = tmpdata.I[i]
                            I_orig = self.problem.IS.inverse_transform(np.array(t, ndmin=2))[0]
                            points1 = {self.problem.IS[k].name: I_orig[k] for k in range(self.problem.DI)}
                            modeldata=[]
                            for p in range(len(tmpdata.P[i])):
                                x = tmpdata.P[i][p]
                                x_orig = self.problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                                points = {self.problem.PS[k].name: x_orig[k] for k in range(self.problem.DP)}
                                points.update(points1)
                                if(tmpdata.D is not None):
                                    points.update(tmpdata.D[i])
                                if(self.problem.constants is not None):
                                    points.update(self.problem.constants)
                                modeldata.append(self.problem.models(points))
                            modeldata=np.array(modeldata)
                            tmpdata.P[i] = np.hstack((tmpdata.P[i],modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space

                    for i in range(len(tmpdata.P)):
                        if(T_sampleflag[i] is False and tmpdata.P[i].shape[0]==0):
                            raise Exception("T_sampleflag[%d] is False but self.data.P[%d] has no data"%(i,i))

                    # print(tmpdata.P[0])
                    #print ("[bestxopt]: len: " + str(len(bestxopt)) + " val: " + str(bestxopt))
                    if (kwargs["model_class"] == "Model_LCM"):
                        # YC: [TODO-check] NSmin checking (with updated NSmin) still necessary? In case of options['model_output_constraint']='Ignore', the updated tmpdata.P and tmpdata.O can have unequal number of samples, but seems like it's running anyways. For now, we can say options['model_output_constraint']='Ignore' is not recommended for Model_LCM mode.
                        for i in range(len(tmpdata.P)):   # LCM requires the same number of samples per task, so use the first NSmin samples
                            tmpdata.O[i] = tmpdata.O[i][0:NSmin,:]
                            tmpdata.P[i] = tmpdata.P[i][0:NSmin,:]

                        (bestxopt, neg_log_marginal_likelihood,
                                gradients, iteration) = \
                            modelers[o].train(data = tmpdata, **kwargs)

                        if evaluation_instance == 0:
                            self.historydb.store_model_LCM(
                                    o,
                                    self.problem,
                                    data_replica.I,
                                    bestxopt,
                                    neg_log_marginal_likelihood,
                                    gradients,
                                    iteration)
                        stats["modeling_iteration"][optiter-1] += iteration
                    else:
                        (hyperparameters, modeling_options, model_stats) = modelers[o].train(data = tmpdata, **kwargs)
                        if evaluation_instance == 0:
                            self.historydb.store_model_GPy_LCM(
                                    o,
                                    self.problem,
                                    data_replica.I,
                                    hyperparameters,
                                    modeling_options,
                                    model_stats)

                    if self.options['verbose'] == True and self.options['model_class'] == 'Model_LCM' and len(data_replica.I)>1:
                        C = modelers[o].M.kern.get_correlation_metric()
                        print("The correlation matrix C is \n", C)
                    elif self.options['verbose'] == True and self.options['model_class'] == 'Model_GPy_LCM' and len(data_replica.I)>1:
                        C = modelers[o].get_correlation_metric(len(data_replica.I))
                        print("The correlation matrix C is \n", C)

                t2 = time.time_ns()
                stats["modeling_time"].append((t2-t1)/1e9)
                time_model = time_model + (t2-t1)/1e9

                t1 = time.time_ns()
                res = searcher.search_multitask(data = data_replica, models = modelers, tids=tids, **kwargs)
                i1=0

                newdata_for_replica = Data(problem = self.problem, I = self.data.I, D = self.data.D)
                newdata_for_replica.P=[]
                for i in range(NI):
                    if(T_sampleflag[i] is True):
                        newdata.P[i].append(res[i1][1][0][0])
                        #newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:] # if NSi>=NS, skip the function evaluation

                        newdata_for_replica.P.append(res[i1][1][0])
                        i1=i1+1
                        NSi = data_replica.P[i].shape[0]
                        newdata_for_replica.P[i] = newdata_for_replica.P[i][0:min(newdata_for_replica.P[i].shape[0],max(0,NS-NSi)),:] # if NSi>=NS, skip the function evaluation
                    else:
                        newdata.P.append(np.empty(shape=(0,self.problem.DP)))
                        newdata_for_replica.P.append(np.empty(shape=(0,self.problem.DP)))

                # print(more_samples,newdata.P)
                t2 = time.time_ns()
                stats["search_time"].append((t2-t1)/1e9)
                time_search = time_search + (t2-t1)/1e9

                newdata_for_replica.O = []
                for i in range(NI):
                    for o in range(self.problem.DO):
                        (mu, var) = modelers[o].predict(np.array(newdata_for_replica.P[i][0]), i)
                        newdata_for_replica.O.append(mu)

                data_replica.merge(newdata_for_replica)

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
            # print(self.data.P)
            # print(list(map(mul,T_bit_mask,list(map(len, self.data.P)))))
            # print(tids)
            NS_active = [list(map(len, self.data.P))[index] for index in tids]
            NSmin = min(NS_active)

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

    def MLA(self, NS, NS1 = None, NI = None, Tgiven = None):

        if NS1 == None:
            NS1 = int(NS/2)
        elif NS1 > NS:
            raise Exception("NS1>NS")

        print("\n\n\n------Starting MLA with %d tasks and %d samples each "%(NI,NS))

        if self.options["BO_objective_evaluation_parallelism"] == False:
            return self.MLA_(NS, NS1, NI, Tgiven, T_sampleflag=[True]*NI, function_evaluations=None, source_function_evaluations=None, models_transfer=None)
        elif self.options["BO_objective_evaluation_parallelism"] == True:
            return self.MLA_ParallelEval_(NS, NS1, NI, Tgiven, T_sampleflag=[True]*NI, function_evaluations=None, source_function_evaluations=None, models_transfer=None)
        else:
            return self.MLA_(NS, NS1, NI, Tgiven, T_sampleflag=[True]*NI, function_evaluations=None, source_function_evaluations=None, models_transfer=None)

        #return self.MLA_(NS, NS1, NI, Tgiven, T_sampleflag=[True]*NI, function_evaluations=None, source_function_evaluations=None, models_transfer=None)

    def TLA(self, NS, Tnew=None, models_transfer=None, source_function_evaluations=None, TLA_options = ["Regression", "LCM", "Stacking"], **kwargs):

        return TLA_I(NS=NS, Tnew=Tnew, models_transfer=models_transfer, source_function_evaluations=source_function_evaluations, TLA_options=TLA_options)

    def TLA_I(self, NS, Tnew=None, models_transfer=None, source_function_evaluations=None, TLA_options = ["Regression", "LCM", "Stacking"], **kwargs):

        # Unified TLA interface
        # This supports only one target task

        if models_transfer == None:
            models_transfer = []

            problem_space = {}
            problem_space["input_space"] = self.historydb.problem_space_to_dict(self.problem.IS)
            problem_space["parameter_space"] = self.historydb.problem_space_to_dict(self.problem.PS)
            problem_space["output_space"] = self.historydb.problem_space_to_dict(self.problem.OS)

            use_LCM_for_source = False

            if use_LCM_for_source == True:
                for i in range(len(source_function_evaluations)):
                    input_task = []

                    input_task_ = []
                    for key in source_function_evaluations[i][0]["task_parameter"]:
                        input_task_.append(source_function_evaluations[i][0]["task_parameter"][key])

                    input_task.append(input_task_)

                    for j in range(len(source_function_evaluations)):
                        if i != j:
                            input_task_ = []
                            for key in source_function_evaluations[j][0]["task_parameter"]:
                                input_task_.append(source_function_evaluations[j][0]["task_parameter"][key])
                            input_task.append(input_task_)

                    function_evaluations_all = []
                    for j in range(len(source_function_evaluations)):
                        function_evaluations_all += source_function_evaluations[j]

                    surrogate_model = BuildSurrogateModel(
                        problem_space = problem_space,
                        modeler = "Model_GPy_LCM",
                        input_task = input_task,
                        function_evaluations = function_evaluations_all)
                        #function_evaluations = source_function_evaluations[i])
                    models_transfer.append(surrogate_model)
            else:
                for i in range(len(source_function_evaluations)):
                    input_task = []
                    for key in source_function_evaluations[i][0]["task_parameter"]:
                        input_task.append(source_function_evaluations[i][0]["task_parameter"][key])

                    surrogate_model = BuildSurrogateModel(
                        problem_space = problem_space,
                        modeler = "Model_GPy_LCM",
                        input_task = [input_task],
                        function_evaluations = source_function_evaluations[i])
                    models_transfer.append(surrogate_model)

        NS1=0
        NI=1
        num_source_tasks = len(models_transfer)
        num_target_tasks = 1

        """ Redefine input space """
        input_space = self.historydb.problem_space_to_dict(self.problem.IS)
        input_space_arr = []
        for input_space_info in input_space:
            name_ = input_space_info["name"]
            type_ = input_space_info["type"]
            transformer_ = input_space_info["transformer"]

            if type_ == "int" or type_ == "Int" or type_ == "Integer" or type_ == "integer":
                lower_bound_ = input_space_info["lower_bound"]
                upper_bound_ = input_space_info["upper_bound"]
                input_space_arr.append(Integer(lower_bound_, upper_bound_, transform=transformer_, name=name_))
            elif type_ == "real" or type_ == "Real" or type_ == "float" or type_ == "Float":
                lower_bound_ = input_space_info["lower_bound"]
                upper_bound_ = input_space_info["upper_bound"]
                input_space_arr.append(Real(lower_bound_, upper_bound_, transform=transformer_, name=name_))
            elif type_ == "categorical" or type_ == "Categorical" or type_ == "category" or type_ == "Category":
                categories = input_space_info["categories"]
                input_space_arr.append(Categoricalnorm(categories, transform=transformer_, name=name_))
        input_space_arr.append(Integer(0, num_source_tasks+num_target_tasks, transform="normalize", name="tla_id"))
        IS = Space(input_space_arr)

        self.tuningproblem.update_input_space(IS)
        self.problem.IS = IS

        """ Redefine the given tasks with source tasks """
        Tnew_ = []
        for i in range(num_target_tasks):
            Tnew_.append(Tnew[i]+[i])

        if self.problem.DO > 1:
            print ("[TLA Warning] currently, TLA does not fully support multi-objective tuning")

        objective_name = self.problem.OS[0].name

        if self.options['TLA_method'] == None:
            print('\n\n\n------Starting Single-task learning (no TLA method option is given) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(models_transfer)))
            return self.MLA_(NS, NS1, NI, Tnew_)

        elif self.options['TLA_method'] == 'Regression':
            print('\n\n\n------Starting TLA (Regression Weighted Sum) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(models_transfer)))
            return self.TLA_Regression(NS, NS1, NI, Tnew_, models_transfer)

        elif self.options['TLA_method'] == 'Sum':
            print('\n\n\n------Starting TLA (Sum) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(models_transfer)))
            return self.TLA_Regression(NS, NS1, NI, Tnew_, models_transfer)

        elif self.options['TLA_method'] == 'LCM_BF':
            print('\n\n\n------Starting TLA (LCM BF) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(models_transfer)))
            Tnew__ = copy.deepcopy(Tnew_)

            for i in range(num_source_tasks):
                task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                task_parameter_vec = []
                for j in range(len(self.problem.IS)):
                    if self.problem.IS[j].name == "tla_id":
                        continue
                    else:
                        task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                task_parameter_vec.append(i+1)
                Tnew__.append(task_parameter_vec)

            return self.TLA_LCM_BF(NS, NS1, NI+num_source_tasks, Tnew__, models_transfer)

        elif self.options['TLA_method'] == 'LCM':
            print('\n\n\n------Starting TLA (LCM GPY) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(models_transfer)))
            Tnew__ = copy.deepcopy(Tnew_)
            for i in range(num_source_tasks):
                task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                task_parameter_vec = []
                for j in range(len(self.problem.IS)):
                    if self.problem.IS[j].name == "tla_id":
                        continue
                    else:
                        task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                task_parameter_vec.append(i+1)
                Tnew__.append(task_parameter_vec)

            NI_ = NI+num_source_tasks
            T_sampleflag = [False]*NI_
            T_sampleflag[0] = True

            return self.MLA_(NS, NS1, NI+num_source_tasks, Tnew__, T_sampleflag, None, source_function_evaluations, models_transfer)

        elif self.options['TLA_method'] == 'Stacking':
            print('\n\n\n------Starting TLA (Stacking) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(source_function_evaluations)))
            return self.TLA_Stacking(NS, NS1, NI, Tnew_, source_function_evaluations, models_transfer)

        elif self.options['TLA_method'] == 'Ensemble_Toggling':
            print('\n\n\n------Starting TLA (Ensemble Toggling) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(source_function_evaluations)))

            for n_sample in range(1, NS+1, 1):
                TLA_chosen = TLA_options[(n_sample-1) % len(TLA_options)]

                # re-initialize the data instance; historydb will load it again
                self.data = Data(self.problem)

                if TLA_chosen == "Regression":
                    self.options["TLA_method"] = "Regression"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "Sum":
                    self.options["TLA_method"] = "Sum"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "LCM_BF":
                    self.options["TLA_method"] = "LCM_BF"

                    Tnew__ = copy.deepcopy(Tnew_)

                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    (data, model, stats) = self.TLA_LCM_BF(n_sample, NS1, NI+num_source_tasks, Tnew_, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)
                elif TLA_chosen == "LCM":
                    self.options["TLA_method"] = "LCM"
                    Tnew__ = copy.deepcopy(Tnew_)
                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    NI_ = NI+num_source_tasks
                    T_sampleflag = [False]*NI_
                    T_sampleflag[0] = True
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI+num_source_tasks, Tnew__, T_sampleflag, None, source_function_evaluations, models_transfer)
                elif TLA_chosen == "Stacking":
                    self.options["TLA_method"] = "Stacking"
                    (data, model, stats) = self.TLA_Stacking(n_sample, NS1, NI, Tnew_, source_function_evaluations, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)

            return (data, model, stats)

        elif self.options['TLA_method'] == 'Ensemble_Peeking':
            print('\n\n\n------Starting TLA (Ensemble Toggling) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(source_function_evaluations)))

            for n_sample in range(1, NS+1, 1):
                self.options['model_peeking_level'] = len(TLA_options)
                TLA_chosen = TLA_options[(n_sample-1) % len(TLA_options)]

                # re-initialize the data instance; historydb will load it again
                self.data = Data(self.problem)

                if TLA_chosen == "Regression":
                    self.options["TLA_method"] = "Regression"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "Sum":
                    self.options["TLA_method"] = "Sum"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "LCM_BF":
                    self.options["TLA_method"] = "LCM_BF"

                    Tnew__ = copy.deepcopy(Tnew_)

                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    (data, model, stats) = self.TLA_LCM_BF(n_sample, NS1, NI+num_source_tasks, Tnew_, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)
                elif TLA_chosen == "LCM":
                    self.options["TLA_method"] = "LCM"
                    Tnew__ = copy.deepcopy(Tnew_)
                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    NI_ = NI+num_source_tasks
                    T_sampleflag = [False]*NI_
                    T_sampleflag[0] = True
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI+num_source_tasks, Tnew__, T_sampleflag, None, source_function_evaluations, models_transfer)
                elif TLA_chosen == "Stacking":
                    self.options["TLA_method"] = "Stacking"
                    (data, model, stats) = self.TLA_Stacking(n_sample, NS1, NI, Tnew_, source_function_evaluations, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)

            return (data, model, stats)

        elif self.options['TLA_method'] == 'Ensemble_Prob':
            print('\n\n\n------Starting TLA (Ensemble Probability) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(source_function_evaluations)))

            exploration_rate = self.options["TLA_ensemble_exploration_rate"]

            best_result = {}
            for TLA_option in TLA_options:
                best_result[TLA_option] = 1.0
            num_TLA_options = len(TLA_options)

            def select_via_probability(best_result):
                option_vec = [k for k in best_result]
                result_vec = np.array([best_result[k] for k in best_result])
                best_min = np.min(result_vec)
                if (best_min <= 0.0):
                    diff = 1 - best_min
                    for i in range(len(result_vec)):
                        result_vec[i] = diff + result_vec[i]
                for i in range(len(result_vec)):
                    result_vec[i] = 1.0/result_vec[i]
                sum_ = np.sum(result_vec)
                for i in range(len(result_vec)):
                    result_vec[i] = result_vec[i]/sum_
                for i in range(1, len(result_vec), 1):
                    result_vec[i] += result_vec[i-1]

                # test
                rand_num1 = np.random.rand()
                if rand_num1 < exploration_rate:
                    option_idx = np.random.choice(num_TLA_options)
                    return option_vec[option_idx]
                else:
                    rand_num = np.random.rand()
                    option_idx = 0
                    for i in range(len(result_vec)):
                        if rand_num <= result_vec[i]:
                            option_idx = i
                            return option_vec[option_idx]

            for n_sample in range(1, NS+1, 1):

                if n_sample <= num_TLA_options:
                    TLA_chosen = TLA_options[(n_sample-1) % len(TLA_options)]
                else:
                    TLA_chosen = select_via_probability(best_result)

                # re-initialize the data instance; historydb will load it again
                self.data = Data(self.problem)

                if TLA_chosen == "Regression":
                    self.options["TLA_method"] = "Regression"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "Sum":
                    self.options["TLA_method"] = "Sum"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "LCM_BF":
                    self.options["TLA_method"] = "LCM_BF"

                    Tnew__ = copy.deepcopy(Tnew_)

                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    (data, model, stats) = self.TLA_LCM_BF(n_sample, NS1, NI+num_source_tasks, Tnew_, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)
                elif TLA_chosen == "LCM":
                    self.options["TLA_method"] = "LCM"
                    Tnew__ = copy.deepcopy(Tnew_)
                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    NI_ = NI+num_source_tasks
                    T_sampleflag = [False]*NI_
                    T_sampleflag[0] = True
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI+num_source_tasks, Tnew__, T_sampleflag, None, source_function_evaluations, models_transfer)
                elif TLA_chosen == "Stacking":
                    self.options["TLA_method"] = "Stacking"
                    (data, model, stats) = self.TLA_Stacking(n_sample, NS1, NI, Tnew_, source_function_evaluations, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)

                # currently hard coded.. it will work only for single target task TLA
                with open(self.historydb.historydb_path+"/"+self.historydb.tuning_problem_name+".json", "r") as f_in:
                    loaded_function_evaluations = json.load(f_in)["func_eval"]

                    for TLA_option in TLA_options:
                        if TLA_option == "Sum":
                            modeling_load = "TLA_Sum"
                        elif TLA_option == "Regression":
                            modeling_load = "TLA_RegressionSum"
                        elif TLA_option == "LCM_BF":
                            modeling_load = "TLA_LCM_BF"
                        elif TLA_option == "LCM":
                            modeling_load = "TLA_LCM"
                        elif TLA_option == "Stacking":
                            modeling_load = "TLA_Stacking"

                        best_result_ = None
                        for i in range(len(loaded_function_evaluations)):
                            func_eval = loaded_function_evaluations[i]
                            if func_eval["modeling"] == modeling_load:
                                if best_result_ == None or func_eval["evaluation_result"][objective_name] < best_result_:
                                    best_result_ = func_eval["evaluation_result"][objective_name]
                                    best_result[TLA_option] = best_result_

            return (data, model, stats)

        elif self.options['TLA_method'] == 'Ensemble_ProbDyn':
            print('\n\n\n------Starting TLA (Ensemble Probability) for %d tasks and %d samples each with %d source tasks '%(NI,NS,len(source_function_evaluations)))

            best_result = {}
            for TLA_option in TLA_options:
                best_result[TLA_option] = 1.0
            num_TLA_options = len(TLA_options)

            def sigmoid(num_samples, num_options, num_parameters):
                x = (num_options*(num_parameters))*1.0/num_samples
                return float(x)/(1+abs(x))
                #return float(x)/math.sqrt(1+x**2)

            def select_via_probability(exploration_rate, best_result):
                option_vec = [k for k in best_result]
                result_vec = np.array([best_result[k] for k in best_result])
                best_min = np.min(result_vec)
                if (best_min <= 0.0):
                    diff = 1 - best_min
                    for i in range(len(result_vec)):
                        result_vec[i] = diff + result_vec[i]
                for i in range(len(result_vec)):
                    result_vec[i] = 1.0/result_vec[i]
                sum_ = np.sum(result_vec)
                for i in range(len(result_vec)):
                    result_vec[i] = result_vec[i]/sum_
                for i in range(1, len(result_vec), 1):
                    result_vec[i] += result_vec[i-1]

                # test
                rand_num1 = np.random.rand()
                if rand_num1 < exploration_rate:
                    option_idx = np.random.choice(num_TLA_options)
                    return option_vec[option_idx]
                else:
                    rand_num = np.random.rand()
                    option_idx = 0
                    for i in range(len(result_vec)):
                        if rand_num <= result_vec[i]:
                            option_idx = i
                            return option_vec[option_idx]

            for n_sample in range(1, NS+1, 1):

                if n_sample <= num_TLA_options:
                    TLA_chosen = TLA_options[(n_sample-1) % len(TLA_options)]
                else:
                    exploration_rate = sigmoid(n_sample-1, len(TLA_options), len(self.problem.PS))
                    TLA_chosen = select_via_probability(exploration_rate, best_result)

                # re-initialize the data instance; historydb will load it again
                self.data = Data(self.problem)

                if TLA_chosen == "Regression":
                    self.options["TLA_method"] = "Regression"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "Sum":
                    self.options["TLA_method"] = "Sum"
                    (data, model, stats) = self.TLA_Regression(n_sample, NS1, NI, Tnew_, models_transfer)
                elif TLA_chosen == "LCM_BF":
                    self.options["TLA_method"] = "LCM_BF"

                    Tnew__ = copy.deepcopy(Tnew_)

                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    (data, model, stats) = self.TLA_LCM_BF(n_sample, NS1, NI+num_source_tasks, Tnew_, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)
                elif TLA_chosen == "LCM":
                    self.options["TLA_method"] = "LCM"
                    Tnew__ = copy.deepcopy(Tnew_)
                    for i in range(num_source_tasks):
                        task_parameter_dict = source_function_evaluations[i][0]["task_parameter"]
                        task_parameter_vec = []
                        for j in range(len(self.problem.IS)):
                            if self.problem.IS[j].name == "tla_id":
                                continue
                            else:
                                task_parameter_vec.append(task_parameter_dict[self.problem.IS[j].name])
                        task_parameter_vec.append(i+1)
                        Tnew__.append(task_parameter_vec)

                    NI_ = NI+num_source_tasks
                    T_sampleflag = [False]*NI_
                    T_sampleflag[0] = True
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI+num_source_tasks, Tnew__, T_sampleflag, None, source_function_evaluations, models_transfer)
                elif TLA_chosen == "Stacking":
                    self.options["TLA_method"] = "Stacking"
                    (data, model, stats) = self.TLA_Stacking(n_sample, NS1, NI, Tnew_, source_function_evaluations, models_transfer)
                elif TLA_chosen == "SLA":
                    self.options["TLA_method"] = None
                    (data, model, stats) = self.MLA_(n_sample, NS1, NI, Tnew_)

                # currently hard coded..
                with open(self.historydb.historydb_path+"/"+self.historydb.tuning_problem_name+".json", "r") as f_in:
                    loaded_function_evaluations = json.load(f_in)["func_eval"]

                    for TLA_option in TLA_options:
                        if TLA_option == "Sum":
                            modeling_load = "TLA_Sum"
                        elif TLA_option == "Regression":
                            modeling_load = "TLA_RegressionSum"
                        elif TLA_option == "LCM_BF":
                            modeling_load = "TLA_LCM_BF"
                        elif TLA_option == "LCM":
                            modeling_load = "TLA_LCM"
                        elif TLA_option == "Stacking":
                            modeling_load = "TLA_Stacking"

                        best_result_ = None
                        for i in range(len(loaded_function_evaluations)):
                            func_eval = loaded_function_evaluations[i]
                            if func_eval["modeling"] == modeling_load:
                                if best_result_ == None or func_eval["evaluation_result"][objective_name] < best_result_:
                                    best_result_ = func_eval["evaluation_result"][objective_name]
                                    best_result[TLA_option] = best_result_

            return (data, model, stats)

    def TLA_Regression(self, NS, NS1 = None, NI = None, Tgiven = None, models_transfer = None, **kwargs):
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

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven)

        print ("self.data.P: ", self.data.P)

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

        """ Multi-task Learning Autotuning """

        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

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

        print ("NS1: ", NS1)
        is_pilot = False
        if NS1 == 0:
            NS1 = 1
            searcher_tla = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options, models_transfer = models_transfer)')
            res = searcher_tla.search_multitask(data = self.data, models = None, **kwargs)
            tmpP = [x[1][0] for x in res]
            #for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
            #    NSi = self.data.P[i].shape[0]
            #    newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]
            ## print(more_samples,newdata.P)


            #tmpP = [(self.problem.PS.transform([[128,24]]))]
            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    NSi = self.data.P[i].shape[0]
                    tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data
        else:
            is_pilot = True
            if (NSmin<NS1):
                check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
                tmpP = sampler.sample_parameters(problem = self.problem, n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
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
            tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs, is_pilot=is_pilot)
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
        self.models_transfer = models_transfer
        searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options, models_transfer = self.models_transfer)')
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
            print("Iteration: ", optiter)
            stats["modeling_iteration"].append(0)
            optiter = optiter + 1
            t1 = time.time_ns()
            for o in range(self.problem.DO):
                #tmpdata = copy.deepcopy(self.data)
                #print ("tmpdata.I: ", tmpdata.I)
                #print ("tmpdata.P: ", tmpdata.P)
                #print ("tmpdata.O: ", tmpdata.O)
                #tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]

                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
                    tmp=[]
                    for x in tmpdata.P:
                        if(len(x)>0):
                            xNorm = self.problem.PS.transform(x)
                            tmp.append(xNorm)
                        else:
                            tmp.append(np.empty( shape=(0, self.problem.DP) ))
                    tmpdata.P=tmp
                if tmpdata.I is not None: # from a list of lists to a 2D numpy array
                    tmpdata.I = self.problem.IS.transform(tmpdata.I)
                #print ("tmpdata.I: ", tmpdata.I)
                #print ("tmpdata.P: ", tmpdata.P)
                #print ("tmpdata.O: ", tmpdata.O)

                if (kwargs["model_output_constraint"] != None):
                    tmp_tmpdata = Data(self.problem)
                    tmp_tmpdata.I = copy.deepcopy(self.data.I)
                    tmp_tmpdata.P = [[] for i in range(len(self.data.P))]
                    tmp_tmpdata.O = [[] for i in range(len(self.data.O))]
                    tmp_tmpdata.D = copy.deepcopy(self.data.D)

                    for t in range(len(tmpdata.O)):
                        for i in range(len(tmpdata.O[t])):
                            out_of_range = False
                            for o_ in range(self.problem.DO):
                                output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                lower_bound = output_space["lower_bound"]
                                upper_bound = output_space["upper_bound"]
                                output_result = [copy.deepcopy(self.data.O[i][:,o_].reshape((-1,1))) for i in range(len(self.data.I))][t][i]
                                if output_result < lower_bound or \
                                   output_result > upper_bound:
                                    out_of_range = True
                            if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                if (kwargs["model_output_constraint"] == 'LargeNum'):
                                    tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                    tmp_tmpdata.O[t].append([1000000000.0]) #sys.float_info.max
                                elif (kwargs["model_output_constraint"] == 'Ignore'):
                                    pass
                            else:
                                tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                tmp_tmpdata.O[t].append(tmpdata.O[t][i])

                    for t in range(len(tmpdata.O)):
                        tmp_tmpdata.P[t] = np.array(tmp_tmpdata.P[t])
                        tmp_tmpdata.O[t] = np.array(tmp_tmpdata.O[t])

                    for t in range(len(tmp_tmpdata.O)):
                        if len(tmp_tmpdata.O[t]) > 0:
                            tmpdata.P[t] = copy.deepcopy(tmp_tmpdata.P[t])
                            tmpdata.O[t] = copy.deepcopy(tmp_tmpdata.O[t])

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
                    (hyperparameters, modeling_options, model_stats) = modelers[o].train(data = tmpdata, **kwargs)
                    self.historydb.store_model_GPy_LCM(
                            o,
                            self.problem,
                            self.data.I,
                            hyperparameters,
                            modeling_options,
                            model_stats)

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

    def TLA_LCM_BF(self, NS, NS1 = None, NI = None, Tgiven = None, models_transfer = None, **kwargs):
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

        print ("Tgiven")
        print (Tgiven)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven) #, function_evaluations, options=kwargs)

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

        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

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
            self.data.O = self.computer.evaluate_objective_TLA(self.problem, self.data.I, self.data.P, self.data.D, self.historydb, options = kwargs, models_transfer=models_transfer)

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

        print ("NS1: ", NS1)
        if NS1 == 0:
            NS1 = 1
            searcher_tla = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options, models_transfer = models_transfer)')
            res = searcher_tla.search_multitask(data = self.data, models = None, **kwargs)
            tmpP = [x[1][0] for x in res]
            #for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
            #    NSi = self.data.P[i].shape[0]
            #    newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]
            ## print(more_samples,newdata.P)


            #tmpP = [(self.problem.PS.transform([[128,24]]))]
            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    NSi = self.data.P[i].shape[0]
                    tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data
        else:
            if (NSmin<NS1):
                check_constraints = functools.partial(self.computer.evaluate_constraints, self.problem, inputs_only = False, kwargs = kwargs)
                tmpP = sampler.sample_parameters(problem = self.problem, n_samples = NS1-NSmin, I = self.data.I, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)
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
            tmpO = self.computer.evaluate_objective_TLA(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs, models_transfer=models_transfer)
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
        self.models_transfer = models_transfer
        searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options, models_transfer = self.models_transfer)')
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
            print("Iteration: ", optiter)
            stats["modeling_iteration"].append(0)
            optiter = optiter + 1
            t1 = time.time_ns()
            for o in range(self.problem.DO):

                tmpdata = copy.deepcopy(self.data)
                tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]

                if (kwargs["model_output_constraint"] != None):
                    tmp_tmpdata = Data(self.problem)
                    tmp_tmpdata.I = copy.deepcopy(self.data.I)
                    tmp_tmpdata.P = [[] for i in range(len(self.data.P))]
                    tmp_tmpdata.O = [[] for i in range(len(self.data.O))]
                    tmp_tmpdata.D = copy.deepcopy(self.data.D)

                    for t in range(len(tmpdata.O)):
                        for i in range(len(tmpdata.O[t])):
                            out_of_range = False
                            for o_ in range(self.problem.DO):
                                output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                lower_bound = output_space["lower_bound"]
                                upper_bound = output_space["upper_bound"]
                                output_result = [copy.deepcopy(self.data.O[i][:,o_].reshape((-1,1))) for i in range(len(self.data.I))][t][i]
                                if output_result < lower_bound or \
                                   output_result > upper_bound:
                                    out_of_range = True
                            if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                if (kwargs["model_output_constraint"] == 'LargeNum'):
                                    tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                    tmp_tmpdata.O[t].append([1000000000.0]) #sys.float_info.max
                                elif (kwargs["model_output_constraint"] == 'Ignore'):
                                    pass
                            else:
                                tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                tmp_tmpdata.O[t].append(tmpdata.O[t][i])

                    for t in range(len(tmpdata.O)):
                        tmp_tmpdata.P[t] = np.array(tmp_tmpdata.P[t])
                        tmp_tmpdata.O[t] = np.array(tmp_tmpdata.O[t])

                    for t in range(len(tmp_tmpdata.O)):
                        if len(tmp_tmpdata.O[t]) > 0:
                            tmpdata.P[t] = copy.deepcopy(tmp_tmpdata.P[t])
                            tmpdata.O[t] = copy.deepcopy(tmp_tmpdata.O[t])

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
                    (hyperparameters, modeling_options, model_stats) = modelers[o].train(data = tmpdata, **kwargs)
                    self.historydb.store_model_GPy_LCM(
                            o,
                            self.problem,
                            self.data.I,
                            hyperparameters,
                            modeling_options,
                            model_stats)

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
            newdata.O = self.computer.evaluate_objective_TLA(problem = self.problem,
                    I = newdata.I,
                    P = newdata.P,
                    D = newdata.D,
                    history_db = self.historydb,
                    options = kwargs,
                    models_transfer = models_transfer)
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

    def TLA_Stacking(self, NS, NS1 = None, NI = None, Tgiven = None, source_function_evaluations = None, models_transfer = None, **kwargs):
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

        options1 = copy.deepcopy(self.options)
        kwargs.update(options1)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven)

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

        """ Multi-task Learning Autotuning """

        if(Tgiven is not None and self.data.I is None):  # building the MLA model for each of the given tasks
            self.data.I = Tgiven

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

        # Build source models
        def load_source_data(problem: Problem, source_function_evaluations):
            data = Data(problem = problem)
            num_tasks = 1 # for now we consider only one task for target task

            PS_history = [[] for i in range(num_tasks)]
            OS_history = [[] for i in range(num_tasks)]

            num_loaded_data = 0

            for func_eval in source_function_evaluations:
                task_id = 0
                parameter_arr = []
                for k in range(len(problem.PS)):
                    if type(problem.PS[k]).__name__ == "Categoricalnorm":
                        parameter_arr.append(str(func_eval["tuning_parameter"][problem.PS[k].name]))
                    elif type(problem.PS[k]).__name__ == "Integer":
                        parameter_arr.append(int(func_eval["tuning_parameter"][problem.PS[k].name]))
                    elif type(problem.PS[k]).__name__ == "Real":
                        parameter_arr.append(float(func_eval["tuning_parameter"][problem.PS[k].name]))
                    else:
                        parameter_arr.append(func_eval["tuning_parameter"][problem.PS[k].name])
                #print ("parameter_arr: ", parameter_arr)
                PS_history[task_id].append(parameter_arr)
                OS_history[task_id].append(\
                    [func_eval["evaluation_result"][problem.OS[k].name] \
                    for k in range(len(problem.OS))])
                num_loaded_data += 1

            print ("Tgiven: ", Tgiven)

            if (num_loaded_data > 0):
                data.I = Tgiven # fake task info (may change to IS_history)
                #data.P = PS_history
                data.P=[]
                for i in range(len(PS_history)):
                    if(len(PS_history[i])==0):
                        data.P.append(np.empty( shape=(0, problem.DO)))
                    else:
                        data.P.append(np.array(PS_history[i]))
                        if(any(ele==[None] for ele in PS_history[i])):
                            print ("history data contains null function values")
                            exit()
                data.O=[] # YL: OS is a list of 2D numpy arrays
                for i in range(len(OS_history)):
                    if(len(OS_history[i])==0):
                        data.O.append(np.empty( shape=(0, problem.DO)))
                    else:
                        data.O.append(np.array(OS_history[i]))
                        if(any(ele==[None] for ele in OS_history[i])):
                            print ("history data contains null function values")
                            exit()

                data.I = problem.IS.transform(data.I)
                if data.P is not None:
                    tmp=[]
                    for x in data.P:
                        if(len(x)>0):
                            xNorm = problem.PS.transform(x)
                            tmp.append(xNorm)
                        else:
                            tmp.append(np.empty( shape=(0, self.problem.DP) ))
                    data.P=tmp

                #print ("db: data.I: " + str(data.I))
                #print ("db: data.P: " + str(data.P))
                #print ("db: data.O: " + str(data.O))
            else:
                print ("no history data has been loaded")
            return data

        searcher = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options)')

        num_source_tasks = len(source_function_evaluations)

        modelers = [eval(f'{kwargs["model_class"]} (problem = self.problem, computer = self.computer)')]*self.problem.DO
        # train initial models
        for o in range(self.problem.DO):
            modelers[o].train_stacked(data = load_source_data(self.problem, source_function_evaluations[0]), num_source_tasks = num_source_tasks, **kwargs)

        def load_residuals(problem: Problem, modeler, current_function_evaluations):
            data = Data(problem=problem)
            num_tasks = 1 # for now we consider only one task for target task

            PS_history = [[] for i in range(num_tasks)]
            OS_history = [[] for i in range(num_tasks)]

            num_loaded_data = 0

            task_parameter_names = [self.problem.IS[k].name for k in range(len(self.problem.IS))]
            tuning_parameter_names = [self.problem.PS[k].name for k in range(len(self.problem.PS))]
            tuning_parameter_types = [type(self.problem.PS[k]).__name__ for k in range(len(self.problem.PS))]
            output_names = [self.problem.OS[k].name for k in range(len(self.problem.OS))]

            tid = 0

            for func_eval in current_function_evaluations:
                task_id = 0
                parameter_arr = []
                for k in range(len(problem.PS)):
                    if type(problem.PS[k]).__name__ == "Categoricalnorm":
                        parameter_arr.append(str(func_eval["tuning_parameter"][problem.PS[k].name]))
                    elif type(problem.PS[k]).__name__ == "Integer":
                        parameter_arr.append(int(func_eval["tuning_parameter"][problem.PS[k].name]))
                    elif type(problem.PS[k]).__name__ == "Real":
                        parameter_arr.append(float(func_eval["tuning_parameter"][problem.PS[k].name]))
                    else:
                        parameter_arr.append(func_eval["tuning_parameter"][problem.PS[k].name])
                PS_history[task_id].append(parameter_arr)

                input_tuning_parameters_transformed = self.problem.PS.transform([parameter_arr])[0]

                residual_result = []
                for k in range(len(problem.OS)):
                    mu, var = modeler.predict(np.array(input_tuning_parameters_transformed),tid)
                    observation = func_eval["evaluation_result"][problem.OS[k].name]
                    print ("mu: ", mu)
                    print ("observation: ", observation)
                    residual = observation - mu[0][0] #- observation
                    residual_result.append(residual)
                OS_history[task_id].append(residual_result)
                num_loaded_data += 1

            if (num_loaded_data > 0):
                data.I = Tgiven #IS_history
                #data.P = PS_history
                data.P=[]
                for i in range(len(PS_history)):
                    if(len(PS_history[i])==0):
                        data.P.append(np.empty( shape=(0, problem.DO)))
                    else:
                        data.P.append(np.array(PS_history[i]))
                        if(any(ele==[None] for ele in PS_history[i])):
                            print ("history data contains null function values")
                            exit()
                data.O=[] # YL: OS is a list of 2D numpy arrays
                for i in range(len(OS_history)):
                    if(len(OS_history[i])==0):
                        data.O.append(np.empty( shape=(0, problem.DO)))
                    else:
                        data.O.append(np.array(OS_history[i]))
                        if(any(ele==[None] for ele in OS_history[i])):
                            print ("history data contains null function values")
                            exit()

                data.I = problem.IS.transform(data.I)
                if data.P is not None:
                    tmp=[]
                    for x in data.P:
                        if(len(x)>0):
                            xNorm = problem.PS.transform(x)
                            tmp.append(xNorm)
                        else:
                            tmp.append(np.empty( shape=(0, self.problem.DP) ))
                    data.P=tmp

                # print ("db: data.I: " + str(data.I))
                # print ("db: data.P: " + str(data.P))
                # print ("db: data.O: " + str(OS_history))
            else:
                print ("no history data has been loaded")
            return data

        # train stacking models based on residuals
        for o in range(self.problem.DO):
            for i in range(1, num_source_tasks, 1):
                modelers[o].train_stacked(data=load_residuals(self.problem, modelers[o], source_function_evaluations[i]), num_source_tasks=num_source_tasks, **kwargs)

        initial_modelers = copy.deepcopy(modelers)

        print ("NS1: ", NS1)
        is_pilot = False
        run_pilot_anyway = False
        if NS1 == 0:
            NS1 = 1
            searcher_tla = eval(f'{kwargs["search_class"]}(problem = self.problem, computer = self.computer, options = self.options, models_transfer = models_transfer)')
            res = searcher_tla.search_multitask(data = self.data, models = None, **kwargs)
            tmpP = [x[1][0] for x in res]
            #for i in range(len(newdata.P)):  # if NSi>=NS, skip the function evaluation
            #    NSi = self.data.P[i].shape[0]
            #    newdata.P[i] = newdata.P[i][0:min(newdata.P[i].shape[0],max(0,NS-NSi)),:]
            ## print(more_samples,newdata.P)

            run_pilot_anyway = False
            if (kwargs["model_input_separation"] == True or kwargs["model_peeking_level"] > 1):
                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is None:
                    # no samples from this modeling approach
                    run_pilot_anyway = True

            #tmpP = [(self.problem.PS.transform([[128,24]]))]
            if(self.data.P is not None):
                for i in range(len(self.data.P)):
                    if run_pilot_anyway == False:
                        NSi = self.data.P[i].shape[0]
                        tmpP[i] = tmpP[i][0:max(NS1-NSi,0),:] # if NSi>=NS1, no need to generate new random data
        else:
            is_pilot = True
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
        if (NSmin<NS1 or run_pilot_anyway == True):
            tmpO = self.computer.evaluate_objective(self.problem, self.data.I, tmpP, self.data.D, self.historydb, options = kwargs, is_pilot=is_pilot)
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
            print("Iteration: ", optiter)
            stats["modeling_iteration"].append(0)
            optiter = optiter + 1
            t1 = time.time_ns()
            for o in range(self.problem.DO):
                #tmpdata = copy.deepcopy(self.data)
                #tmpdata.O = [copy.deepcopy(self.data.O[i][:,o].reshape((-1,1))) for i in range(len(self.data.I))]

                tmpdata = Data(self.problem)
                self.historydb.load_history_func_eval(tmpdata, self.problem, Tgiven, options=kwargs)
                if tmpdata.P is not None: # from a list of (list of lists) to a list of 2D numpy arrays
                    tmp=[]
                    for x in tmpdata.P:
                        if(len(x)>0):
                            xNorm = self.problem.PS.transform(x)
                            tmp.append(xNorm)
                        else:
                            tmp.append(np.empty( shape=(0, self.problem.DP) ))
                    tmpdata.P=tmp
                if tmpdata.I is not None: # from a list of lists to a 2D numpy array
                    tmpdata.I = self.problem.IS.transform(tmpdata.I)
                print ("tmpdata.I: ", tmpdata.I)
                print ("tmpdata.P: ", tmpdata.P)
                print ("tmpdata.O: ", tmpdata.O)

                if (kwargs["model_output_constraint"] != None):
                    tmp_tmpdata = Data(self.problem)
                    tmp_tmpdata.I = copy.deepcopy(self.data.I)
                    tmp_tmpdata.P = [[] for i in range(len(self.data.P))]
                    tmp_tmpdata.O = [[] for i in range(len(self.data.O))]
                    tmp_tmpdata.D = copy.deepcopy(self.data.D)

                    for t in range(len(tmpdata.O)):
                        for i in range(len(tmpdata.O[t])):
                            out_of_range = False
                            for o_ in range(self.problem.DO):
                                output_space = self.historydb.problem_space_to_dict(self.problem.OS)[o_]
                                lower_bound = output_space["lower_bound"]
                                upper_bound = output_space["upper_bound"]
                                output_result = [copy.deepcopy(self.data.O[i][:,o_].reshape((-1,1))) for i in range(len(self.data.I))][t][i]
                                if output_result < lower_bound or \
                                   output_result > upper_bound:
                                    out_of_range = True
                            if out_of_range == True or self.historydb.problem_space_to_dict(self.problem.OS)[o]["optimize"] == False:
                                if (kwargs["model_output_constraint"] == 'LargeNum'):
                                    tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                    tmp_tmpdata.O[t].append([1000000000.0]) #sys.float_info.max
                                elif (kwargs["model_output_constraint"] == 'Ignore'):
                                    pass
                            else:
                                tmp_tmpdata.P[t].append(tmpdata.P[t][i])
                                tmp_tmpdata.O[t].append(tmpdata.O[t][i])

                    for t in range(len(tmpdata.O)):
                        tmp_tmpdata.P[t] = np.array(tmp_tmpdata.P[t])
                        tmp_tmpdata.O[t] = np.array(tmp_tmpdata.O[t])

                    for t in range(len(tmp_tmpdata.O)):
                        if len(tmp_tmpdata.O[t]) > 0:
                            tmpdata.P[t] = copy.deepcopy(tmp_tmpdata.P[t])
                            tmpdata.O[t] = copy.deepcopy(tmp_tmpdata.O[t])

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

                def data_to_dict(problem, data):
                    data_ = copy.deepcopy(data)
                    if data_.I is not None:    # from 2D numpy array to a list of lists
                        data_.I = problem.IS.inverse_transform(data_.I)
                    if data_.P is not None:    # from a collection of 2D numpy arrays to a list of (list of lists)
                        data_.P = [problem.PS.inverse_transform(x) for x in data_.P]

                    task_id = 0
                    documents = []

                    num_evals = len(data_.P[task_id])
                    print ("num_evals: ", num_evals)

                    for i in range(num_evals):
                        document = {}
                        document["task_parameter"] = { problem.IS[k].name:data_.I[task_id][k] for k in range(len(problem.IS)) }
                        document["tuning_parameter"] = { problem.PS[k].name:data_.P[task_id][i][k] for k in range(len(problem.PS)) }
                        document["evaluation_result"] = { problem.OS[k].name:data_.O[task_id][i][k] for k in range(len(problem.OS)) }

                        print ("document: ", document)

                        documents.append(document)

                    return documents
                modelers[o].train_stacked(data = load_residuals(self.problem, initial_modelers[o], data_to_dict(self.problem, self.data)), num_source_tasks=num_source_tasks, **kwargs)

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

    def TLA_II(self, Tnew, Tsrc = None, source_function_evaluations = None):

        print('\n\n\n------Starting TLA_II for task: ',Tnew)

        if Tsrc == None and source_function_evaluations == None:
            raise Exception("No historical data is given for TLA_II")

        if Tsrc == None and source_function_evaluations != None:
            Tsrc = []
            for source_task_id in range(len(source_function_evaluations)):
                source_task_parameter = []
                for key in source_function_evaluations[source_task_id][0]["task_parameter"]:
                    source_task_parameter.append(source_function_evaluations[source_task_id][0]["task_parameter"][key])
                Tsrc.append(source_task_parameter)

        """ Load history function evaluation data """
        if self.historydb.load_func_eval == True:
            # load function evaluations regardless of the modeling scheme of the sample
            self.historydb.load_history_func_eval(self.data, self.problem, Tgiven=Tsrc, function_evaluations=None, source_function_evaluations=source_function_evaluations, options=None)

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
            raise Exception("TLA_II only works for single-objective tuning")

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
            M.optimize_restarts(num_restarts = kwargs['model_restarts'], robust=True, verbose = kwargs['verbose'], parallel = (kwargs['model_threads'] > 1), num_processes = kwargs['model_threads'], messages = kwargs['verbose'], optimizer = kwargs['model_optimizer'], start = None, max_iters = kwargs['model_max_iters'], ipython_notebook = False, clear_after_finish = True)
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
        #        pickle.dump(aprxopts, open('TLA_II.pkl', 'w'))

        t4 = time.time_ns()
        stats['time_total'] = (t4-t3)/1e9
        stats['time_fun'] = time_fun

        return (aprxopts, O, stats)



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


    def MB_LCM(self, NLOOP=None, Tgiven=None, Pdefault=None, **kwargs):
        """
        Tgiven		 : a list of tasks 
        NLOOP	     : number of GPTuneBand loops 
        Pdefault     : assuming there is a default parameter configuration among all tasks
        """

        np.set_printoptions(suppress=False, precision=4)
        print('\n\n\n------Starting MB_LCM (multi-arm bandit with LCM) with %d loops for task' % (NS), Tgiven)

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
        # print(int((self.smax+1)/(0+1))*self.options['budget_base']**0,self.smax,'jifdjfd')
        NSs1 = [0] * len(self.NSs)
        info = [[x, y] for x, y in zip(self.budgets, self.NSs)]
        print('total samples:', info)
                
        self.data.I = Tgiven
        # self.data.D = [] 
        self.data.P = []
        self.data.O = []
        
        data = Data(self.tp)   # data contains all data in the extended task space returned by MLA

        for Nloop in range(NLOOP):
            data1 = Data(self.tp)  # for each loop, data1 will have all data for all arms sampled by MLA (excluding SH samples)

            newtasks = []
            if(self.data.D is not None):
                data1.D = []
            for s1 in range(0, len(self.budgets)):
                for t in range(len(Tgiven)):
                    budget1 = self.budgets[s1]
                    tmp = [budget1]+Tgiven[t]
                    newtasks.append(tmp)
                    if(self.data.D is not None):
                        if(len(self.data.D)>0):
                            data1.D.append(self.data.D[t])
            if Nloop == 0:
                all_subtasks = copy.deepcopy(newtasks)
            data1.I = newtasks


            for s in range(len(self.budgets)):  # loop over the budget levels
                budget = self.budgets[s]
                ns = self.NSs[s] 
                ntotal = NSs1[s] + int(ns)
                NSs1[s] = NSs1[s] + ns
                print(f"Bracket s = {s}, budget = {budget}, ns = {ns}")
                if(Pdefault is not None):
                    data1.P = [[Pdefault]] * len(newtasks) # as gt.MLA will load available database, Pdefault is effective only if database is empty   
                
                # print("Calling MLA: \ndata1.I", data1.I, "\ndata1.P", data1.P, "\ndata1.O", data1.O)
                # print(f"NS={ntotal}, Tgiven={newtasks}, NI={len(newtasks)}, NS1={min(self.NSs)}")
                gt = GPTune(self.tp, computer=self.computer,
                            data=data1, options=self.options)
                T_sampleflag = [False] * (len(newtasks))
                idx = s*len(Tgiven) 
                T_sampleflag[idx:] = [True]*(len(newtasks)-idx)
                # print(newtasks)
                # print(T_sampleflag)

                (data1, _, stats0) = gt.MLA_(NS=ntotal, Tgiven=newtasks, NI=len(newtasks), NS1=min(self.NSs), T_sampleflag=T_sampleflag)
                
                # print("Finish Calling MLA: \ndata1.I", data1.I, "\ndata1.P", data1.P, "\ndata1.O", data1.O)
                # merge new results to history
                
                stats['time_total'] += stats0['time_total']
                stats['time_fun'] += stats0['time_fun']
                stats['time_model'] += stats0['time_model']
                stats['time_search'] += stats0['time_search']
                stats['time_sample_init'] += stats0['time_sample_init']


            # bug fix: data1 will load all available data from database, the following make srue that it only contains MLA samples generated at the current loop    
            for s in range(len(self.budgets)):  # loop over the budget levels
                idx = s*len(Tgiven) 
                for i in range(len(Tgiven)):
                    data1.P[idx+i] = data1.P[idx+i][NSs1[s] - self.NSs[s]:NSs1[s]]
                    data1.O[idx+i] = data1.O[idx+i][NSs1[s] - self.NSs[s]:NSs1[s],:]

            # print('MLA samples generated at current loop:')
            # print('data1.I: ', data1.I)
            # print('data1.P: ', data1.P)
            # print('data1.O: ', data1.O)
            # print("data1.D = ", data1.D)

            if Nloop == 0:
                self.data.P = data1.P[0:len(Tgiven)]  # the first 0:len(Tgiven) tasks of data1 are highest budget
                self.data.O = data1.O[0:len(Tgiven)] 
            else:
                # self.data.P = [np.concatenate((self.data.P[i], data1.P[0:len(Tgiven)][i])) for i in range(len(self.data.P))]
                self.data.P = [self.data.P[i] + data1.P[i] for i in range(len(self.data.P))]
                self.data.O = [np.concatenate((self.data.O[i], data1.O[i])) for i in range(len(self.data.O))]
            
            print("Finish multi-arm initial evaluation")
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
                idx = s*len(Tgiven) 
                temp_I = data1.I[idx:idx+len(Tgiven)]
                # print(f'Tasks: ', temp_I)
                temp_O = list(map(np.squeeze, data1.O[idx:idx+len(Tgiven)]))
                # temp_P = list(map(np.array, data1.P[idx:idx+len(Tgiven)]))
                temp_P = data1.P[idx:idx+len(Tgiven)]
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

                    gt.historydb.load_history_func_eval(newdata, gt.problem, temp_I)


                    t1 = time.time_ns()
                    done=0
                    # print("load_history_func_eval in SH", gt.data.P, NSs1[s-ri-1]+ratio)
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
                        newdata.O = gt.computer.evaluate_objective(gt.problem, gt.data.I, gt.data.P, gt.data.D, gt.historydb, options = kwargs)
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
                newdata.I = Tgiven
                # print('newdata before merge')
                # print('newdata.P = ', newdata.P)
                # print('self.data.P = ', self.data.P)
                # self.data.merge(newdata) # this would make self.data.P a list of numpy array
                self.data.P = [x + y for x, y in zip(self.data.P, newdata.P)]
                self.data.O = [np.concatenate((self.data.O[i], newdata.O[i])) for i in range(len(self.data.O))]
                
                # print('Data updated: ')
                # print('self.data.P = ', self.data.P)
                # print('self.data.O = ', self.data.O)
            
            print('Updated self.data (only highest fidelity) after all SH runs')
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
            print('data1_hist (including MLA samples for all fidelities) ')    
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

def BuildSurrogateModel(problem_space:dict=None, modeler:str="Model_GPy_LCM", input_task:list=[], function_evaluations:list=None):

    input_space_info = problem_space["input_space"]
    parameter_space_info = problem_space["parameter_space"]
    output_space_info = problem_space["output_space"]

    input_space_arr = []
    for input_space_info in problem_space["input_space"]:
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
    for parameter_space_info in problem_space["parameter_space"]:
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
    for output_space_info in problem_space["output_space"]:
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
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1
    options['model_processes'] = 1
    options['model_class'] = modeler
    options['verbose'] = False
    options.validate(computer=computer)
    historydb = HistoryDB(meta_dict=problem_space)
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb)
    (models, model_function) = gt.GenSurrogateModel(input_task, function_evaluations)

    return (model_function)

def PredictOutput(problem_space:dict=None,
        modeler:str="Model_GPy_LCM",
        input_task:list=[],
        input_parameter:dict={},
        surrogate_model=None,
        function_evaluations=None):

    if surrogate_model == None:
        surrogate_model = BuildSurrogateModel(problem_space = problem_space,
                modeler = modeler,
                input_task = [input_task],
                function_evaluations = function_evaluations)

    ret = surrogate_model(point = input_parameter)

    return ret

def SensitivityAnalysis(
        problem_space:dict=None,
        modeler:str="Model_GPy_LCM",
        method="Sobol",
        input_task:list=[],
        surrogate_model=None,
        function_evaluations=None,
        historydb_path:str=None,
        tuning_problem_name:str=None,
        num_samples:int=1000):

    # Prepare the surrogate model for running sensitivity analysis
    if surrogate_model == None:
        if function_evaluations == None:
            if historydb_path != None:
                with open(historydb_path, "r") as f_in:
                    function_evaluations = json.load(f_in)["func_eval"]
                surrogate_model = BuildSurrogateModel(problem_space = problem_space,
                        modeler = modeler,
                        input_task = [input_task],
                        function_evaluations = function_evaluations)
            elif tuning_problem_name != None:
                with open("gptune.db/"+tuning_problem_name+".json", "r") as f_in:
                    function_evaluations = json.load(f_in)["func_eval"]
                surrogate_model = BuildSurrogateModel(problem_space = problem_space,
                        modeler = modeler,
                        input_task = [input_task],
                        function_evaluations = function_evaluations)
            else:
                print ("no data is given, and cannot build a surrogate performance model, exit..")
                exit()
        else:
            surrogate_model = BuildSurrogateModel(problem_space = problem_space,
                    modeler = modeler,
                    input_task = [input_task],
                    function_evaluations = function_evaluations)

    if method == "Sobol":
        from SALib.sample import saltelli
        from SALib.analyze import sobol

        num_vars = len(problem_space["parameter_space"])
        parameter_names = []
        parameter_bounds = []
        parameter_types = []
        categorical_parameter_values = {}
        for parameter_info in problem_space["parameter_space"]:
            parameter_name = parameter_info["name"]
            parameter_type = parameter_info["type"]

            parameter_names.append(parameter_name)
            parameter_types.append(parameter_type)

            if parameter_type == "int" or parameter_type == "integer" or parameter_type == "real":
                lower_bound = parameter_info["lower_bound"]
                upper_bound = parameter_info["upper_bound"]
                parameter_bounds.append([lower_bound, upper_bound])
            elif parameter_type == "categorical":
                lower_bound = 0
                upper_bound = len(parameter_info["categories"])
                parameter_bounds.append([lower_bound, upper_bound])
                categorical_parameter_values[parameter_name] = parameter_info["categories"]

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
                if parameter_types[j] == "int" or parameter_types[j] == "integer":
                    model_input[parameter_names[j]] = int(parameter_values[i][j])
                elif parameter_types[j] == "real":
                    model_input[parameter_names[j]] = parameter_values[i][j]
                elif parameter_types[j] == "categorical":
                    model_input[parameter_names[j]] = categorical_parameter_values[parameter_names[j]][int(parameter_values[i][j])]

            num_task_parameters = len(problem_space["input_space"])
            for j in range(num_task_parameters):
                model_input[problem_space["input_space"][j]["name"]] = input_task[j]
            #print (model_input)

            output_name = problem_space["output_space"][0]["name"]

            Y.append(surrogate_model(model_input)[output_name][0][0])

        # Sensitivity analysis based on the samples drawn from the fitted surrogate model.
        Si = sobol.analyze(problem, np.array(Y), print_to_console=False)

        ret = {}

        S1 = {}
        for i in range(len(parameter_names)):
            S1[parameter_names[i]] = Si["S1"][i]
        ret["S1"] = S1

        S1_conf = {}
        for i in range(len(parameter_names)):
            S1_conf[parameter_names[i]] = Si["S1_conf"][i]
        ret["S1_conf"] = S1_conf

        S2 = {}
        for i in range(len(parameter_names)):
            S2[parameter_names[i]] = {}
            for j in range(len(parameter_names)):
                S2[parameter_names[i]][parameter_names[j]] =Si["S2"][i][j]
            #Si["S2"][i]
        ret["S2"] = S2

        S2_conf = {}
        for i in range(len(parameter_names)):
            S2_conf[parameter_names[i]] = {}
            for j in range(len(parameter_names)):
                S2_conf[parameter_names[i]][parameter_names[j]] =Si["S2_conf"][i][j]
        ret["S2_conf"] = S2_conf

        ST = {}
        for i in range(len(parameter_names)):
            ST[parameter_names[i]] = Si["ST"][i]
        ret["ST"] = ST

        ST_conf = {}
        for i in range(len(parameter_names)):
            ST_conf[parameter_names[i]] = Si["ST_conf"][i]
        ret["ST_conf"] = ST_conf

        return ret

    else:
        return

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

