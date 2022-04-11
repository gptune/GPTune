#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import os

if __name__ == "__main__":

    target = "H2O"

    experiment_name = "superlu_tuning_space_reduction_" + target

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10,4)) #, constrained_layout=True)

    tuning_all_10th = 0
    tuning_all_20th = 0
    tuning_reduced_default_10th = 0
    tuning_reduced_default_20th = 0

    for plot_num in [0,1,2,3]:
        ax = axs[plot_num]

        for tuner in ["tuning_all", "tuning_reduced_default"]: #, "tuning_reduced_random"]:
            batches_num_func_eval = []
            batches_best_tuning_result = []

            for batch_num in [0,1,2]:
                search_logfile = "gptune.db/SuperLU_DIST-pddrive_spawn-"+tuner+"-"+target+".mtx-npilot0-nbatch"+str(batch_num)+".json"
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

                    for i in range(0, len(function_evaluations), 1):
                        func_eval = function_evaluations[i]
                        runtime = func_eval["evaluation_result"]["runtime"]

                        if best_runtime == None or runtime < best_runtime:
                            best_runtime = runtime

                        num_func_eval.append(i+1)
                        best_tuning_result.append(best_runtime)

                    print ("num_func_eval: ", num_func_eval)
                    print ("best_tuning_result: ", best_tuning_result)

                    if plot_num == 0:
                        point_list = [i for i in range(1, 6, 1)]
                    elif plot_num == 1:
                        point_list = [i for i in range(6, 11, 1)]
                    elif plot_num == 2:
                        point_list = [i for i in range(11, 16, 1)]
                    elif plot_num == 3:
                        point_list = [i for i in range(16, 21, 1)]

                    num_func_eval = [num_func_eval[i-1] for i in point_list]
                    best_tuning_result = [best_tuning_result[i-1] for i in point_list]
                    print ("num_func_eval: ", num_func_eval)
                    print ("best_tuning_result: ", best_tuning_result)

                    batches_num_func_eval.append(num_func_eval)
                    batches_best_tuning_result.append(best_tuning_result)

            # plotting
            if len(batches_num_func_eval) >= 1:
                plot_start = 0
                print ("target: ", target)
                print ("tuner: ", tuner)
                print ("plot_start: ", plot_start)

                num_func_eval = batches_num_func_eval[0]
                #best_tuning_result = np.average(batches_best_tuning_result) #, axis=0)
                print ("batches_best_tuning_result: ", batches_best_tuning_result)
                best_tuning_result = np.mean(batches_best_tuning_result, axis=0)
                print ("tuner: " ,tuner)
                print ("num_func_eval: ", num_func_eval)
                print ("best_tuning_result: ", best_tuning_result)

                if plot_num == 1:
                    if tuner == "tuning_all":
                        tuning_all_10th = round(best_tuning_result[-1],2)
                    elif tuner == "tuning_reduced_default":
                        tuning_reduced_default_10th = round(best_tuning_result[-1],2)
                elif plot_num == 3:
                    if tuner == "tuning_all":
                        tuning_all_20th = round(best_tuning_result[-1],2)
                    elif tuner == "tuning_reduced_default":
                        tuning_reduced_default_20th = round(best_tuning_result[-1],2)

                #print ("tuner: ", tuner, " best_tuning_result_10th:", best_tuning_result[9])
                #print ("tuner: ", tuner, " best_tuning_result_5th:", best_tuning_result[4])
                solution = round(best_tuning_result[-1],3)
                print ("solution: ", solution)
                best_tuning_result_lower = np.std(batches_best_tuning_result, axis=0)
                best_tuning_result_upper = np.std(batches_best_tuning_result, axis=0)
                print ("npstd: ", np.std(batches_best_tuning_result, axis=0))

                if tuner == "tuning_all":
                    #label_name = "Tuning five parameters (" + str(solution) +")"
                    label_name = "Original tuning problem (five parameters)"
                    if plot_num == 3:
                        label_name = "Original tuning problem (five parameters), 10th result: " + str(tuning_all_10th) + "s, 20th result: " + str(tuning_all_20th)+"s"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:blue', label=label_name)
                elif tuner == "tuning_reduced_default":
                    #label_name = "Tuning three parameters (" + str(solution) +")"
                    label_name = "Tuning problem reduction (three parameters)"
                    if plot_num == 3:
                        label_name = "Reduced tuning problem (three parameters), 10th result: " + str(tuning_reduced_default_10th) + "s, 20th result: " + str(tuning_reduced_default_20th)+"s"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:orange', label=label_name)
                elif tuner == "tuning_reduced_random":
                    #label_name = "Tuning three parameters (random) (" + str(solution) +")"
                    label_name = "Tuning three parameters (random)"
                    ax.errorbar(num_func_eval, best_tuning_result, yerr=np.std(batches_best_tuning_result, axis=0), marker='o', color='tab:green', label=label_name)

        if plot_num == 3:
            leg = ax.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True, framealpha=0.5, bbox_to_anchor=(1.15, 1.45))
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1.0)

        ax.set_xticks(list(np.arange(plot_num*5+1, (plot_num+1)*5+1)))
        ax.set_xticklabels(list(np.arange(plot_num*5+1, (plot_num+1)*5+1)), fontsize=16)

        if plot_num == 3:
            #ax.set_xlim([16,20])
            ax.set_ylim([3.45,3.65])
            ax.set_yticks([3.5,3.6])
            ax.set_yticklabels([3.5,3.6], fontsize=16)

        ax.tick_params(labelsize=16)

    fig.suptitle("Tuning on SuperLU_DIST (matrix: " + target + ")", fontsize=20)
    fig.supxlabel("Number of function evaluations", fontsize=20)
    fig.supylabel("Best tuning result (time (s))", fontsize=20)
    fig.subplots_adjust(top=0.7, bottom=0.2, wspace=0.33)
    fig.tight_layout()
    fig.savefig(experiment_name+".pdf")
