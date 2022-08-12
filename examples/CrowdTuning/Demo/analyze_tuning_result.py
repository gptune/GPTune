#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import os

if __name__ == "__main__":

    experiment_name = "demo_tuning"

    #for task in [1.01, 1.02, 1.03, 1.04, 1.1, 1.2]:
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10)) #, constrained_layout=True)
    for plot_num in [0,1,2,3]:
        if plot_num == 0:
            task = 1.01
            ax = axs[0,0]
        elif plot_num == 1:
            task = 1.02
            ax = axs[0,1]
        elif plot_num == 2:
            task = 1.1
            ax = axs[1,0]
        elif plot_num == 3:
            task = 1.2
            ax = axs[1,1]

        for tuner in ["SLA", "TLA_Sum", "TLA_Regression", "TLA_LCM_BF", "TLA_LCM_GPY"]:
            batches_num_func_eval = []
            batches_best_tuning_result = []

            for batch_num in [0,1,2,3,4]:
                search_logfile = "gptune.db/GPTune-Demo-"+tuner+"-"+str(task)+"-"+str(batch_num)+"-npilot0.json"

                if not os.path.exists(search_logfile):
                    print ("file not found: ", search_logfile)
                    continue
                else:
                    print ("file found: ", search_logfile)

                with open(search_logfile, "r") as f_in:
                    print ("tuner: ", tuner, "batch: ", batch_num, "search_logfile: ", search_logfile)
                    num_func_eval = []
                    best_tuning_result = []

                    function_evaluations = []
                    for func_eval in json.load(f_in)["func_eval"]:
                        if func_eval["task_parameter"]["t"] == task:
                            function_evaluations.append(func_eval)

                    #function_evaluations = json.load(f_in)["func_eval"]

                    best_y = None

                    for i in range(0, len(function_evaluations), 1):
                        func_eval = function_evaluations[i]

                        x = func_eval["tuning_parameter"]["x"]
                        y = func_eval["evaluation_result"]["y"]

                        if best_y == None or y < best_y:
                            best_y = y

                        num_func_eval.append(i+1)
                        best_tuning_result.append(best_y)

                    print ("num_func_eval: ", num_func_eval)
                    print ("best_tuning_result: ", best_tuning_result)

                    ##point_list = [i for i in range(10, 201, 10)]
                    #point_list = [i for i in range(2, 21, 2)]
                    #num_func_eval = [num_func_eval[i-1] for i in point_list]
                    #best_tuning_result = [best_tuning_result[i-1] for i in point_list]
                    #print ("num_func_eval: ", num_func_eval)
                    #print ("best_tuning_result: ", best_tuning_result)
                    ##best_tuning_annotate = str(i)

                    batches_num_func_eval.append(num_func_eval)
                    batches_best_tuning_result.append(best_tuning_result)

            # plotting
            num_func_eval = batches_num_func_eval[0]
            best_tuning_result = np.mean(batches_best_tuning_result, axis=0)
            print ("tuner: " ,tuner)
            print ("num_func_eval: ", num_func_eval)
            print ("best_tuning_result: ", best_tuning_result)
            print ("task: ", task, " tuner: ", tuner, " best_tuning_result_10th:", best_tuning_result[9])
            solution_10th = round(best_tuning_result[9],3)
            solution_20th = round(best_tuning_result[19],3)
            solution = round(best_tuning_result[-1],3)
            print ("solution: ", solution)
            best_tuning_result_lower = np.std(batches_best_tuning_result, axis=0)
            best_tuning_result_upper = np.std(batches_best_tuning_result, axis=0)
            print ("npstd: ", np.std(batches_best_tuning_result, axis=0))

            if tuner == "SLA":
                label_name = "SLA (" + str(solution) +")"
                label_name = "SLA (10th: " + str(solution_10th) +", 20th: "+str(solution_20th)+ ")"
                ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:blue', label=label_name)
            elif tuner == "TLA_Sum":
                label_name = "TLA: Naive Sum (" + str(solution) +")"
                label_name = "TLA: Naive Sum (10th: " + str(solution_10th) +", 20th: "+str(solution_20th)+ ")"
                ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:red', label=label_name)
            elif tuner == "TLA_Regression":
                label_name = "TLA: Regression Sum (" + str(solution) +")"
                label_name = "TLA: Regression Sum (10th: " + str(solution_10th) +", 20th: "+str(solution_20th)+ ")"
                ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:orange', label=label_name)
            elif tuner == "TLA_LCM_BF":
                label_name = "TLA: LCM BF (" + str(solution) +")"
                label_name = "TLA: LCM BF (10th: " + str(solution_10th) +", 20th: "+str(solution_20th)+ ")"
                ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:purple', label=label_name)
            elif tuner == "TLA_LCM_GPY":
                label_name = "TLA: LCM HD (" + str(solution) +")"
                label_name = "TLA: LCM Refined (10th: " + str(solution_10th) +", 20th: "+str(solution_20th)+ ")"
                ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:green', label=label_name)
            #plt.plot(num_func_eval, best_tuning_result, marker='o', label="GPTune search")

        #plt.title("Tuning NIMROD with 3D SuperLU on 64 Haswell nodes \n Problem size: {mx: 5, my: 7, lphi: 1} \n Source for TLA: 500 samples obtained on 32 nodes")
        ax.set_title("Target task = "+str(task), fontsize=20)
        leg = ax.legend(loc='upper right', fontsize=12, frameon=True)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.0)
        if plot_num == 2 or plot_num == 3:
            ax.set_xlabel("Number of function evaluations", fontsize=20)
        if plot_num == 0 or plot_num == 2:
            ax.set_ylabel("Best tuning result (y)", fontsize=20)
        ax.set_xticks([2,4,6,8,10,12,14,16,18,20]) #, fontsize=12)
        ax.set_xticklabels([2,4,6,8,10,12,14,16,18,20], fontsize=16)
        ax.set_yticks([0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]) #, fontsize=12)
        ax.set_yticklabels([0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5], fontsize=16)
        #ax.set_yticklabels(fontsize=16)
        #ax.set_xticklabels(fontsize=12)

    fig.suptitle("Tuning the demo function \n Source for TLA: 100 samples of task = 1.0", fontsize=24)
    fig.tight_layout()
    fig.savefig(experiment_name+".pdf")
