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

class HistoryDB(dict):

    def __init__(self, **kwargs):

        """ Options for the history database """
        self.history_db = 0
        self.history_db_path = "./"
        self.application_name = None

        """ Pass machine-related information """
        self.machine_name = 'Unknown'
        self.nodes = 'Unknown'
        self.cores = 'Unknown'
        self.nprocmin_pernode = 'Unknown'

        """ Pass software-related information as dictionaries """
        self.compile_deps = {}
        self.runtime_deps = {}

    def load_db(self, data : Data, problem : Problem):

        """ Init history database JSON file """
        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
            if os.path.exists(json_data_path):
                # Load previous history data
                # [TODO] Need to deal with new problems not in the history database
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
                    data.I = IS_history

                    PS_history = []
                    OS_history = []
                    for t in range(num_tasks):
                        PS_history_t = []
                        OS_history_t = []
                        num_evals = len(history_data["perf_data"][t]["func_eval"])
                        for i in range(num_evals):
                            func_eval = history_data["perf_data"][t]["func_eval"][i]
                            PS_history_t.append(\
                                    [func_eval["P"][problem.PS[k].name] \
                                    for k in range(len(problem.PS))])
                            OS_history_t.append(\
                                    [func_eval["O"][problem.OS[k].name] \
                                    for k in range(len(problem.OS))])
                        PS_history.append(PS_history_t)
                        OS_history.append(OS_history_t)
                    data.P = PS_history
                    data.O = np.array(OS_history)
            else:
                print ("Create a JSON file at " + json_data_path)
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
                with open(json_data_path, "w") as f_out:
                    json_data = {}
                    json_data["name"] = self.application_name
                    json_data["perf_data"] = []
                    json.dump(json_data, f_out, indent=2)

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

            with open(json_data_path, "w") as f_out:
                json.dump(json_data, f_out, indent=2)

        return


    def update_func_eval(self, problem : Problem,\
            task_parameter : np.ndarray,\
            tuning_parameter : np.ndarray,\
            evaluation_result : np.ndarray):

        if (self.history_db == 1 and self.application_name is not None):
            json_data_path = self.history_db_path+self.application_name+".json"
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

                #print ("eval: " + str(i))
                #print ("task_parameter: " + str(task_parameter_orig_list))
                #print ("tuning_parameter: " + str(tuning_parameter_orig_list))
                #print ("evaluation_result: " + str(evaluation_result_orig_list))

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
                        "P_m":{'machine':self.machine_name,
                            'nodes':self.nodes,
                            'cores':self.cores,
                            'nprocmin_pernode':self.nprocmin_pernode},
                        "P_s":{'compile_deps':self.compile_deps,
                            'runtime_deps':self.runtime_deps},
                        "O":{problem.OS[k].name:evaluation_result_orig_list[k]
                            for k in range(len(problem.OS))}
                    })

            with open(json_data_path, "w") as f_out:
                json.dump(json_data, f_out, indent=2)

        return

