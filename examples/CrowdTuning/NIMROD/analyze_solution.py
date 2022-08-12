#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import os

if __name__ == "__main__":

    experiment_name = "nimrod_slu3d_tuning_small_and_big_task"

    #for task in [1.01, 1.02, 1.03, 1.04, 1.1, 1.2]:
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,16)) #, constrained_layout=True)
    for plot_num in [0,1,2,3]:
        if plot_num == 0:
            target = "medium_task_haswell_64nodes_2048mpis"
            ax = axs[0]

            #default_parameter_runtimes = [11.3465, 11.7757, 11.9217]
            default_parameter_runtimes = []
            for i in range(3):
                with open(str(target)+"/gptune.db/NIMROD_slu3d_default_parameter_"+str(i)+"_npilot1.json", "r") as f_in:
                    function_evaluation = json.load(f_in)["func_eval"][0]
                    default_parameter_runtime = function_evaluation["evaluation_result"]["time"]
                    default_parameter_runtimes.append(default_parameter_runtime)
            default_parameter_runtime_avg = np.average(default_parameter_runtimes)
            print ("default_runtime_avg: ", default_parameter_runtime_avg)
            ax.axhline(y=default_parameter_runtime_avg, color="red", linestyle="-", label="Default choice (" + str(round(default_parameter_runtime_avg,2))+")")
        elif plot_num == 1:
            target = "medium_task_knl_64nodes_2048mpis"
            ax = axs[1]
            #default_parameter_runtimes = [54.754, 54.8002, 54.6459]
            default_parameter_runtimes = []
            for i in range(3):
                with open(str(target)+"/gptune.db/NIMROD_slu3d_default_parameter_"+str(i)+"_npilot1.json", "r") as f_in:
                    function_evaluation = json.load(f_in)["func_eval"][0]
                    default_parameter_runtime = function_evaluation["evaluation_result"]["time"]
                    default_parameter_runtimes.append(default_parameter_runtime)
            default_parameter_runtime_avg = np.average(default_parameter_runtimes)
            print ("default_runtime_avg: ", default_parameter_runtime_avg)
            ax.axhline(y=default_parameter_runtime_avg, color="red", linestyle="-", label="Default choice (" + str(round(default_parameter_runtime_avg,2))+")")
        elif plot_num == 2:
            target = "medium_task_knl_64nodes_4352mpis"
            ax = axs[2]
            #default_parameter_runtimes = [54.2946, 54.1073, 54.7423]
            default_parameter_runtimes = []
            for i in range(3):
                with open(str(target)+"/gptune.db/NIMROD_slu3d_default_parameter_"+str(i)+"_npilot1.json", "r") as f_in:
                    function_evaluation = json.load(f_in)["func_eval"][0]
                    default_parameter_runtime = function_evaluation["evaluation_result"]["time"]
                    default_parameter_runtimes.append(default_parameter_runtime)
            default_parameter_runtime_avg = np.average(default_parameter_runtimes)
            print ("default_runtime_avg: ", default_parameter_runtime_avg)
            ax.axhline(y=default_parameter_runtime_avg, color="red", linestyle="-", label="Default choice (" + str(round(default_parameter_runtime_avg,2))+")")
        elif plot_num == 3:
            target = "big_task_haswell_64nodes_2048mpis"
            ax = axs[3]
            #default_parameter_runtimes = [66.4903, 65.2984, 65.985]
            default_parameter_runtimes = []
            for i in range(3):
                with open(str(target)+"/gptune.db/NIMROD_slu3d_default_parameter_"+str(i)+"_npilot1.json", "r") as f_in:
                    function_evaluation = json.load(f_in)["func_eval"][0]
                    default_parameter_runtime = function_evaluation["evaluation_result"]["time"]
                    default_parameter_runtimes.append(default_parameter_runtime)
            default_parameter_runtime_avg = np.average(default_parameter_runtimes)
            print ("default_runtime_avg: ", default_parameter_runtime_avg)
            ax.axhline(y=default_parameter_runtime_avg, color="red", linestyle="-", label="Default choice (" + str(round(default_parameter_runtime_avg,2))+")")


        #`for tuner in ["SLA", "SLA-10", "TLA_Regression", "TLA_LCM"]:
        for tuner in ["SLA", "TLA_Regression", "TLA_LCM"]:
        #for tuner in ["SLA", "TLA_Regression", "TLA_LCM"]:
            batches_num_func_eval = []
            batches_best_tuning_result = []

            for batch_num in [0,1,2]:
                if tuner == "SLA-10":
                    search_logfile = str(target)+"/gptune.db/NIMROD_slu3d_SLA_"+str(batch_num)+"_npilot10.json"
                else:
                    search_logfile = str(target)+"/gptune.db/NIMROD_slu3d_"+tuner+"_"+str(batch_num)+"_npilot0.json"

                if not os.path.exists(search_logfile):
                    print ("file not found: ", search_logfile)
                    continue
                else:
                    print ("file found: ", search_logfile)

                with open(search_logfile, "r") as f_in:
                    print ("tuner: ", tuner, "batch: ", batch_num, "search_logfile: ", search_logfile)
                    num_func_eval = []
                    best_tuning_result = []

                    function_evaluations = json.load(f_in)["func_eval"]

                    best_runtime = None

                    failure_flag = False

                    for i in range(0, len(function_evaluations), 1):
                        func_eval = function_evaluations[i]
                        runtime = func_eval["evaluation_result"]["time"]

                        if best_runtime == None or runtime < best_runtime:
                            best_runtime = runtime

                        #if runtime == 1000 or runtime == 500:
                        #    print ("point failure: log: ", search_logfile + " idx: ", i, " runtime: ", runtime)
                        #    failure_flag = True

                        num_func_eval.append(i+1)
                        best_tuning_result.append(best_runtime)

                    print ("num_func_eval: ", num_func_eval)
                    print ("best_tuning_result: ", best_tuning_result)

                    ###point_list = [i for i in range(10, 201, 10)]
                    #point_list = [i for i in range(2, 21, 2)]
                    #num_func_eval = [num_func_eval[i-1] for i in point_list]
                    #best_tuning_result = [best_tuning_result[i-1] for i in point_list]
                    #print ("num_func_eval: ", num_func_eval)
                    #print ("best_tuning_result: ", best_tuning_result)
                    #best_tuning_annotate = str(i)

                    if failure_flag == False:
                        batches_num_func_eval.append(num_func_eval)
                        batches_best_tuning_result.append(best_tuning_result)

            # plotting
            if len(batches_num_func_eval) >= 1:
                plot_start = 0
                for num_evals in range(0, 20, 1):
                    for batch in [0,1,2]:
                        if batches_best_tuning_result[batch][num_evals] == 1000 or batches_best_tuning_result[batch][num_evals] == 500:
                            plot_start = num_evals+1
                print ("target: ", target)
                print ("tuner: ", tuner)
                print ("plot_start: ", plot_start)

                if plot_start != 0:
                    new_batches_num_func_eval = []
                    new_batches_best_tuning_result = []

                    for batch in [0,1,2]:
                        point_list = [i for i in range(plot_start, 20, 1)]
                        num_func_eval = [batches_num_func_eval[batch][i] for i in point_list]
                        best_tuning_result = [batches_best_tuning_result[batch][i] for i in point_list]
                        print ("num_func_eval: ", num_func_eval)
                        print ("best_tuning_result: ", best_tuning_result)
                        new_batches_num_func_eval.append(num_func_eval)
                        new_batches_best_tuning_result.append(best_tuning_result)

                    batches_num_func_eval = new_batches_num_func_eval
                    batches_best_tuning_result = new_batches_best_tuning_result

                num_func_eval = batches_num_func_eval[0]
                #best_tuning_result = np.average(batches_best_tuning_result) #, axis=0)
                print ("batches_best_tuning_result: ", batches_best_tuning_result)
                best_tuning_result = np.mean(batches_best_tuning_result, axis=0)
                print ("tuner: " ,tuner)
                print ("num_func_eval: ", num_func_eval)
                print ("best_tuning_result: ", best_tuning_result)
                print ("tuner: ", tuner, " best_tuning_result_10th:", best_tuning_result[9])
                print ("tuner: ", tuner, " best_tuning_result_5th:", best_tuning_result[4])
                solution = round(best_tuning_result[-1],2)
                print ("solution: ", solution)
                best_tuning_result_lower = np.std(batches_best_tuning_result, axis=0)
                best_tuning_result_upper = np.std(batches_best_tuning_result, axis=0)
                print ("npstd: ", np.std(batches_best_tuning_result, axis=0))

                if tuner == "SLA":
                    label_name = "SLA (" + str(solution) +")"
                    label_name = "SLA (10th: " + str(round(best_tuning_result[9-plot_start],2)) + ", 20th: " + str(round(best_tuning_result[19-plot_start],2)) + ")"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:blue', label=label_name)
                elif tuner == "SLA-10":
                    label_name = "SLA (10 random samples) (" + str(solution) +")"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='black', label=label_name)
                elif tuner == "TLA_Sum":
                    label_name = "TLA: Naive Sum (" + str(solution) +")"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:red', label=label_name)
                elif tuner == "TLA_Regression":
                    label_name = "TLA: Regression Sum (" + str(solution) +")"
                    label_name = "TLA: Regression Sum (10th: " + str(round(best_tuning_result[9-plot_start],2)) + ", 20th: " + str(round(best_tuning_result[19-plot_start],2)) + ")"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:orange', label=label_name)
                #elif tuner == "TLA_LCM_BF":
                #    label_name = "TLA: LCM BF (" + str(solution) +")"
                #    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:purple', label=label_name)
                elif tuner == "TLA_LCM":
                    label_name = "TLA: LCM HD (" + str(solution) +")"
                    label_name = "TLA: LCM Refined (10th: " + str(round(best_tuning_result[9-plot_start],2)) + ", 20th: " + str(round(best_tuning_result[19-plot_start],2)) + ")"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:green', label=label_name)
                #plt.plot(num_func_eval, best_tuning_result, marker='o', label="GPTune search")

        if target == "medium_task_haswell_64nodes_2048mpis":
            ax.set_title("(a) {mx:5, my:7, lphi:1}, 64 Haswell nodes, 2048 MPIs", fontsize=20)
        elif target == "medium_task_knl_64nodes_2048mpis":
            ax.set_title("(b) {mx:5, my:7, lphi:1}, 64 KNL nodes, 2048 MPIs", fontsize=20)
        elif target == "medium_task_knl_64nodes_4352mpis":
            ax.set_title("(c) {mx:5, my:7, lphi:1}, 64 KNL nodes, 4352 MPIs", fontsize=20)
        elif target == "big_task_haswell_64nodes_2048mpis":
            ax.set_title("(d) {mx:6, my:8, lphi:1}, 64 Haswell nodes, 2048 MPIs", fontsize=20)
        leg = ax.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True, framealpha=0.5)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.0)
        ax.set_xlabel("Number of function evaluations", fontsize=20)
        ax.set_ylabel("Best tuning result (time (s))", fontsize=20)
        #ax.set_xticks(list(np.arange(1,21,1)))
        ax.set_xticks([2,4,6,8,10,12,14,16,18,20]) #, fontsize=12)
        ax.set_xticklabels([2,4,6,8,10,12,14,16,18,20], fontsize=16)

        if target == "medium_task_haswell_64nodes_2048mpis":
            ax.set_yticks([9,10,11,12,13,14])
            ax.set_yticklabels([9,10,11,12,13,14], fontsize=16)
        elif target == "medium_task_knl_64nodes_2048mpis":
            ax.set_ylim([35,60])
            ax.set_yticks([35,40,45,50,55,60])
            ax.set_yticklabels([35,40,45,50,55,60], fontsize=16)
        elif target == "medium_task_knl_64nodes_4352mpis":
            ax.set_ylim([35,60])
            ax.set_yticks([35,40,45,50,55,60])
            ax.set_yticklabels([35,40,45,50,55,60], fontsize=16)
        elif target == "big_task_haswell_64nodes_2048mpis":
            ax.set_ylim([30,140])
            ax.set_yticks(list(np.arange(20,160,20)))
            ax.set_yticklabels(list(np.arange(20,160,20)), fontsize=16)
            #ax.set_yticks([30,35,40,45,50,55,60])
            #ax.set_yticklabels([30,35,40,45,50,55,60], fontsize=16)

    fig.suptitle("Tuning on NIMROD using 3D SuperLU_DIST \n (database source for TLA: 500 samples on {mx:5, my:7, lphi:1}, \n 32 Haswell nodes, 2048 MPIs)\n", fontsize=20)
    fig.tight_layout()
    fig.savefig(experiment_name+".pdf")
