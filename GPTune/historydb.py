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

import numpy as np
from problem import Problem
from data import Data
import json
import os.path
from filelock import FileLock
from autotune.space import *
from autotune.problem import TuningProblem
import uuid

class HistoryDB(dict):

    def __init__(self, **kwargs):

        """ Options for the history database """
        self.history_db = 0
        self.history_db_path = "./"
        self.application_name = None

        """ Pass machine-related information """
        self.machine_deps = {
                    "machine":"Unknown",
                    "nodes":"Unknown",
                    "cores":"Unknown"
                }

        """ Pass software-related information as dictionaries """
        self.compile_deps = {}
        self.runtime_deps = {}

        """ Pass load options """
        self.load_deps = {
                    "machine_deps":{},
                    "software_deps":{
                        "compile_deps":{},
                        "runtime_deps":{}
                    }
                }

        self.verbose_history_db = 1

        """ list of UID of function evaluation data """
        self.uids = []

    def check_load_deps(self, func_eval):
        ''' check machine dependencies '''
        machine_deps = self.load_deps['machine_deps']
        machine_parameter = func_eval['machine_deps']

        ''' check machine configuration dependencies '''
        for dep_name in machine_deps:
            if not machine_parameter[dep_name] in machine_deps[dep_name]:
                print (dep_name+": " + machine_parameter[dep_name] +
                       " is not in load_deps: " + str(machine_deps[dep_name]))
                return False

        ''' check compile-level software dependencies '''
        compile_deps = self.load_deps['software_deps']['compile_deps']
        compile_parameter = func_eval['compile_deps']
        for dep_name in compile_deps.keys():
            deps_passed = False
            for option in range(len(compile_deps[dep_name])):
                software_name = compile_deps[dep_name][option]['name']
                if software_name in compile_parameter.keys():
                    version_split = compile_parameter[software_name]['version_split']
                    version_value = version_split[0]*100+version_split[1]*10+version_split[2]
                    #print ("software_name: " + software_name + " version_value: " + str(version_value))

                    if 'version' in compile_deps[dep_name][option].keys():
                        version_dep_split = compile_deps[dep_name][option]['version']
                        version_dep_value = version_dep_split[0]*100+version_dep_split[1]*10+version_dep_split[2]

                        if version_dep_value == version_value:
                            deps_passed = True

                    if 'version_from' in compile_deps[dep_name][option].keys() and \
                       'version_to' not in compile_deps[dep_name][option].keys():
                        version_dep_from_split = compile_deps[dep_name][option]['version_from']
                        version_dep_from_value = version_dep_from_split[0]*100+version_dep_from_split[1]*10+version_dep_from_split[2]

                        if version_dep_from_value <= version_value:
                            deps_passed = True

                    if 'version_from' not in compile_deps[dep_name][option].keys() and \
                       'version_to' in compile_deps[dep_name][option].keys():
                        version_dep_to_split = compile_deps[dep_name][option]['version_to']
                        version_dep_to_value = version_dep_to_split[0]*100+version_dep_to_split[1]*10+version_dep_to_split[2]

                        if version_dep_to_value >= version_value:
                            deps_passed = True

                    if 'version_from' in compile_deps[dep_name][option].keys() and \
                       'version_to' in compile_deps[dep_name][option].keys():
                        version_dep_from_split = compile_deps[dep_name][option]['version_from']
                        version_dep_from_value = version_dep_from_split[0]*100+version_dep_from_split[1]*10+version_dep_from_split[2]

                        version_dep_to_split = compile_deps[dep_name][option]['version_to']
                        version_dep_to_value = version_dep_to_split[0]*100+version_dep_to_split[1]*10+version_dep_to_split[2]

                        if version_dep_from_value <= version_value and \
                           version_dep_to_value >= version_value:
                            deps_passed = True

            if (deps_passed == False):
                if (self.verbose_history_db):
                    print ("deps_passed failed: " + str(option) + " " + str(software_name))
                return False

        # not yet consider runtime-level software dependencies yet
        runtime_deps = self.load_deps['software_deps']['runtime_deps']

        return True

    def load_db(self, data : Data, problem : Problem):

        """ Init history database JSON file """
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                # Load previous history data
                # [TODO] Need to deal with new problems not in the history database
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        #self.data = Data(self.problem)

                        print ("Found a history database")
                        history_data = json.load(f_in)

                        num_tasks = len(history_data["perf_data"])
                        IS_history = []
                        for t in range(num_tasks):
                            input_dict = history_data["perf_data"][t]["I"]
                            IS_history.append(\
                                    np.array([input_dict[problem.IS[k].name] \
                                    for k in range(len(problem.IS))]))

                        num_loaded_data = 0
                        PS_history = []
                        OS_history = []
                        for t in range(num_tasks):
                            PS_history_t = []
                            OS_history_t = []
                            num_evals = len(history_data["perf_data"][t]["func_eval"])
                            for i in range(num_evals):
                                func_eval = history_data["perf_data"][t]["func_eval"][i]
                                if (self.check_load_deps(func_eval)):
                                    PS_history_t.append(\
                                            [func_eval["P"][problem.PS[k].name] \
                                            for k in range(len(problem.PS))])
                                    OS_history_t.append(\
                                            [func_eval["O"][problem.OS[k].name] \
                                            for k in range(len(problem.OS))])
                                    num_loaded_data += 1
                                else:
                                    print ("failed to load")
                            PS_history.append(PS_history_t)
                            OS_history.append(OS_history_t)

                        # [TODO] quick implementation to avoid setting data class
                        # if no data has been loaded
                        # otherwise, that leads to a problem in gptune.py (line 125)
                        if (num_loaded_data > 0):
                            data.I = IS_history
                            data.P = PS_history
                            data.O = np.array(OS_history)
                            #print ("data.I: " + str(data.I))
                            #print ("data.P: " + str(data.P))
                            #print ("data.O: " + str(data.O))
                        else:
                            print ("no prev data has been loaded")
            else:
                print ("Create a JSON file at " + json_data_path)
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "w") as f_out:
                        json_data = {}
                        json_data["name"] = self.application_name
                        json_data["model_data"] = []
                        json_data["perf_data"] = []

                        json.dump(json_data, f_out, indent=2)

    def update_IS(self, problem : Problem, I : np.ndarray = None):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if not os.path.exists(json_data_path):
                print ("Create a JSON file at " + json_data_path)
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "w") as f_out:
                        json_data = {}
                        json_data["name"] = self.application_name
                        json_data["perf_data"] = []
                        json.dump(json_data, f_out, indent=2)

            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "r") as f_in:
                    json_data = json.load(f_in)

                    O = []
                    num_tasks = len(I)
                    for i in range(num_tasks):
                        t = I[i]
                        I_orig = problem.IS.inverse_transform(np.array(t, ndmin=2))[0]
                        I_orig_list = np.array(I_orig).tolist()

                        input_exist = False
                        for k in range(len(json_data["perf_data"])):
                            compare_all_elems = True
                            for l in range(len(problem.IS)):
                                name = problem.IS[l].name
                                if (json_data["perf_data"][k]["I"][problem.IS[l].name] != I_orig_list[l]):
                                    compare_all_elems = False
                                    break

                            if compare_all_elems == True:
                                #print ("input task already exists")
                                input_exist = True
                                break

                        if input_exist == False:
                            json_data["perf_data"].append({
                                    "I":{problem.IS[k].name:I_orig_list[k] for k in range(len(problem.IS))},
                                    "func_eval":[]
                                    })

            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "w") as f_out:
                    json.dump(json_data, f_out, indent=2)

        return

    def update_func_eval(self, problem : Problem,\
            task_parameter : np.ndarray,\
            tuning_parameter : np.ndarray,\
            evaluation_result : np.ndarray):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "r") as f_in:
                    json_data = json.load(f_in)

            # transform to the original parameter space
            task_parameter_orig = problem.IS.inverse_transform(np.array(task_parameter, ndmin=2))[0]
            task_parameter_orig_list = np.array(task_parameter_orig).tolist()

            num_evals = len(tuning_parameter)
            for i in range(num_evals):
                uid = uuid.uuid1()
                self.uids.append(str(uid))

                tuning_parameter_orig = problem.PS.inverse_transform(
                        np.array(tuning_parameter[i], ndmin=2))[0]
                tuning_parameter_orig_list = np.array(tuning_parameter_orig).tolist()
                evaluation_result_orig_list = np.array(evaluation_result[i]).tolist()

                # find pointer to the json entry for the task parameter
                json_task_idx = 0
                for k in range(len(json_data["perf_data"])):
                    compare_all_elems = True
                    for l in range(len(problem.IS)):
                        name = problem.IS[l].name
                        if (json_data["perf_data"][k]["I"][problem.IS[l].name]
                                != task_parameter_orig_list[l]):
                            compare_all_elems = False
                            break
                    if compare_all_elems == True:
                        json_task_idx = k

                json_data["perf_data"][json_task_idx]["func_eval"].append({
                        "P":{problem.PS[k].name:tuning_parameter_orig_list[k]
                            for k in range(len(problem.PS))},
                        "machine_deps":self.machine_deps,
                        "compile_deps":self.compile_deps,
                        "runtime_deps":self.runtime_deps,
                        "O":{problem.OS[k].name:evaluation_result_orig_list[k]
                            for k in range(len(problem.OS))},
                        "uid":str(uid)
                    })

            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "w") as f_out:
                    json.dump(json_data, f_out, indent=2)

        return

    def is_model_problem_match(self, model_data : dict, tuningproblem : TuningProblem, input_given : np.ndarray):
        model_task_parameters = model_data["task_parameters"]
        input_task_parameters = input_given #np.array(problem.IS.inverse_transform(np.array(input_given, ndmin=2))).tolist()
        #print ("model_task_parameters: ", model_task_parameters)
        #print ("input_task_parameters: ", input_task_parameters)
        if len(model_task_parameters) != len(input_task_parameters):
            return False
        num_tasks = len(input_task_parameters)
        for i in range(num_tasks):
            if len(model_task_parameters[i]) != len(input_task_parameters[i]):
                return False
            for j in range(len(input_task_parameters[i])):
                if model_task_parameters[i][j] != input_task_parameters[i][j]:
                    return False

        IS_model = model_data["problem_space"]["IS"]
        IS_given = self.problem_space_to_dict(tuningproblem.input_space)
        if len(IS_model) != len(IS_given):
            return False
        for i in range(len(IS_given)):
            if IS_model[i]["lower_bound"] != IS_given[i]["lower_bound"]:
                return False
            if IS_model[i]["upper_bound"] != IS_given[i]["upper_bound"]:
                return False
            if IS_model[i]["type"] != IS_given[i]["type"]:
                return False

        PS_model = model_data["problem_space"]["PS"]
        PS_given = self.problem_space_to_dict(tuningproblem.parameter_space)
        if len(PS_model) != len(PS_given):
            return False
        for i in range(len(PS_given)):
            if PS_model[i]["lower_bound"] != PS_given[i]["lower_bound"]:
                return False
            if PS_model[i]["upper_bound"] != PS_given[i]["upper_bound"]:
                return False
            if PS_model[i]["type"] != PS_given[i]["type"]:
                return False

        OS_model = model_data["problem_space"]["OS"]
        OS_given = self.problem_space_to_dict(tuningproblem.output_space)
        if len(OS_model) != len(OS_given):
            return False
        for i in range(len(OS_given)):
            if OS_model[i]["lower_bound"] != OS_given[i]["lower_bound"]:
                return False
            if OS_model[i]["upper_bound"] != OS_given[i]["upper_bound"]:
                return False
            if OS_model[i]["type"] != OS_given[i]["type"]:
                return False

        return True

    def read_model_data(self, tuningproblem=None, Igiven=None, modeler="LCM"):
        ret = []
        print ("problem ", tuningproblem)
        print ("problem input_space ", self.problem_space_to_dict(tuningproblem.input_space))

        if tuningproblem == "None" or Igiven == "None":
            return ret

        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        history_data = json.load(f_in)
                        num_models = len(history_data["model_data"])

                        max_evals = 0
                        max_evals_index = -1 # TODO: if no model is found?
                        for i in range(num_models):
                            model_data = history_data["model_data"][i]
                            if (self.is_model_problem_match(model_data, tuningproblem, Igiven)):
                                ret.append(model_data)

        return ret

    def load_MLE_model_hyperparameters(self, tuningproblem : TuningProblem, input_given : np.ndarray, objective : int):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        history_data = json.load(f_in)

                        max_mle = -9999
                        max_mle_index = -1
                        for i in range(len(history_data["model_data"])):
                            model_data = history_data["model_data"][i]
                            if (self.is_model_problem_match(model_data, tuningproblem, input_given)):
                                log_likelihood = -(model_data["neg_log_likelihood"])
                                if log_likelihood > max_mle:
                                    max_mle = log_likelihood
                                    max_mle_index = i

                        hyperparameters =\
                                history_data["model_data"][max_mle_index]["hyperparameters"]

        return hyperparameters

    def load_AIC_model_hyperparameters(self, tuningproblem : TuningProblem, input_given : np.ndarray, objective : int):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        print ("Found a history database")
                        history_data = json.load(f_in)

                        min_aic = 99999
                        min_aic_index = -1
                        for i in range(len(history_data["model_data"])):
                            model_data = history_data["model_data"][i]
                            if (self.is_model_problem_match(model_data, tuningproblem, input_given)):
                                log_likelihood = -(model_data["neg_log_likelihood"])
                                num_parameters = len(model_data["hyperparameters"])
                                AIC = -1.0 * 2.0 * log_likelihood + 2.0 * num_parameters
                                if AIC < min_aic:
                                    min_aic = AIC
                                    min_aic_index = i

                        hyperparameters =\
                                history_data["model_data"][min_aic_index]["hyperparameters"]

        return hyperparameters

    def load_BIC_model_hyperparameters(self, tuningproblem : TuningProblem, input_given : np.ndarray, objective : int):
        import math

        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        history_data = json.load(f_in)

                        min_bic = 99999
                        min_bic_index = -1
                        for i in range(len(history_data["model_data"])):
                            model_data = history_data["model_data"][i]
                            if (self.is_model_problem_match(model_data, tuningproblem, input_given)):
                                if model_data["objective_id"] == objective:
                                    log_likelihood = -(model_data["neg_log_likelihood"])
                                    num_parameters = len(model_data["hyperparameters"])
                                    num_samples = len(model_data["func_eval"])
                                    BIC = -1.0 * 2.0 * log_likelihood + num_parameters * math.log(num_samples)
                                    if BIC < min_bic:
                                        min_bic = BIC
                                        min_bic_index = i

                        hyperparameters =\
                                history_data["model_data"][min_bic_index]["hyperparameters"]

        return hyperparameters

    def load_max_evals_hyperparameters(self, tuningproblem : TuningProblem, input_given : np.ndarray, objective : int):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        print ("Found a history database")
                        history_data = json.load(f_in)

                        max_evals = 0
                        max_evals_index = -1 # TODO: if no model is found?
                        for i in range(len(history_data["model_data"])):
                            model_data = history_data["model_data"][i]
                            if (self.is_model_problem_match(model_data, tuningproblem, input_given)):
                                num_evals = len(history_data["model_data"][i]["func_eval"])
                                print ("i: " + str(i) + " num_evals: " + str(num_evals))
                                if history_data["model_data"][i]["objective_id"] == objective:
                                    if num_evals > max_evals:
                                        max_evals = num_evals
                                        max_evals_index = i
                        hyperparameters =\
                                history_data["model_data"][max_evals_index]["hyperparameters"]

        return hyperparameters

    def load_model_hyperparameters(self, model_uid):
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "r") as f_in:
                        print ("Found a history database")
                        history_data = json.load(f_in)
                        model_data = history_data["model_data"]
                        num_models = len(model_data)
                        for i in range(num_models):
                            if model_data[i]["uid"] == model_uid:
                                return model_data[i]["hyperparameters"]

        return []

    def problem_space_to_dict(self, space : Space):
        dict_arr = []

        space_len = len(space)

        for i in range(space_len):
            dict_ = {}

            lower_bound, upper_bound = space.bounds[i]

            dict_["lower_bound"] = lower_bound
            dict_["upper_bound"] = upper_bound

            if space.is_real == True:
                dict_["type"] = "real"
            elif space.is_categorical == True:
                dict_["type"] = "categorical"
            else:
                dict_["type"] = "int"

            dict_arr.append(dict_)

        return dict_arr

    def update_model_LCM(self,\
            objective : int,
            problem : Problem,\
            input_given : np.ndarray,\
            bestxopt : np.ndarray,\
            neg_log_marginal_likelihood : float,\
            gradients : np.ndarray,\
            iteration : int):

        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "r") as f_in:
                    json_data = json.load(f_in)

            #self.problem_space_to_dict(problem.IS)

            from scipy.stats.mstats import gmean
            from scipy.stats.mstats import hmean
            model_stats = {}
            model_stats["neg_log_likelihood"] = neg_log_marginal_likelihood
            model_stats["gradients"] = gradients.tolist()
            model_stats["gradients_sum_abs"] = np.sum(np.absolute(gradients))
            model_stats["gradients_average_abs"] = np.average(np.absolute(gradients))
            model_stats["gradients_hmean_abs"] = hmean(np.absolute(gradients))
            model_stats["gradients_gmean_abs"] = gmean(np.absolute(gradients))
            model_stats["iteration"] = iteration

            gradients_list = gradients.tolist()

            problem_space = {}
            problem_space["IS"] = self.problem_space_to_dict(problem.IS)
            problem_space["PS"] = self.problem_space_to_dict(problem.PS)
            problem_space["OS"] = self.problem_space_to_dict(problem.OS)

            task_parameter_orig = problem.IS.inverse_transform(np.array(input_given, ndmin=2))
            task_parameter_orig_list = np.array(task_parameter_orig).tolist()

            json_data["model_data"].append({
                    "hyperparameters":bestxopt.tolist(),
                    "model_stats":model_stats,
                    "iteration":iteration,
                    "func_eval":self.uids,
                    "task_parameters":task_parameter_orig_list,
                    "problem_space":problem_space,
                    "modeler":"Model_LCM",
                    "objective_id":objective,
                    "uid":str(uuid.uuid1())
                    # objective id is to dinstinguish between different models for multi-objective optimization;
                    # we might need a nicer way to manage different models
                })

            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "w") as f_out:
                    json.dump(json_data, f_out, indent=2)

        return

