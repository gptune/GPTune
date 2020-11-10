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
        #self.load_db = 0
        self.load_deps = {
                    "machine_deps":{},
                    "software_deps":{
                        "compile_deps":{},
                        "runtime_deps":{}
                    }
                }

        self.verbose_history_db = 1

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
                            print ("data.O: " + str(data.O))
                        else:
                            print ("no prev data has been loaded")
            else:
                print ("Create a JSON file at " + json_data_path)
                with FileLock(json_data_path+".lock"):
                    with open(json_data_path, "w") as f_out:
                        json_data = {}
                        json_data["name"] = self.application_name
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
                                print ("input task already exists")
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
                            for k in range(len(problem.OS))}
                    })

            with FileLock(json_data_path+".lock"):
                with open(json_data_path, "w") as f_out:
                    json.dump(json_data, f_out, indent=2)

        return

