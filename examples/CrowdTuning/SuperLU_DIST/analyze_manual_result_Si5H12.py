#! /usr/bin/env python

import sys
import os
#sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
#from gptune import *
import json
import numpy as np

def search_func_eval(function_evaluations, point):
    for func_eval in function_evaluations:
        if func_eval["task_parameter"]["matrix"] == point["matrix"] and \
           func_eval["tuning_parameter"]["COLPERM"] == point["COLPERM"] and\
           func_eval["tuning_parameter"]["LOOKAHEAD"] == point["LOOKAHEAD"] and\
           func_eval["tuning_parameter"]["nprows"] == point["nprows"] and\
           func_eval["tuning_parameter"]["NSUP"] == point["NSUP"] and\
           func_eval["tuning_parameter"]["NREL"] == point["NREL"]:
               return func_eval
    return None

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #for matrix in ["Si2.mtx", "SiH4.mtx", "SiNa.mtx", "benzene.mtx", "Si5H12.mtx"]: #Si10H16.mtx SiO.mtx H2O.mtx GaAsH6.mtx Ga3As3H12.mtx
    for matrix in ["Si5H12.mtx"]: #, "Si10H16.mtx", "SiO.mtx", "H2O.mtx", "GaAsH6.mtx", "Ga3As3H12.mtx"]:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,8)) #, constrained_layout=True)
        #axs[-1, -1].axis('off')
        axs[2, 1].axis('off')
        for plot_cnt in range(0, 5, 1):
            with open("SuperLU_DIST-pddrive_spawn-manual_analysis.json", "r") as f_in:
                print (f_in)
                function_evaluations = json.load(f_in)
                print (function_evaluations)

                x_observed = []
                y_observed = []

                COLPERM = '4'
                LOOKAHEAD = 10
                nprows = 8
                NSUP = 128
                NREL = 20

                if plot_cnt == 0:
                    for COLPERM in ['1','2','3','4','5']:
                        point = {
                            "matrix": matrix,
                            "COLPERM": COLPERM,
                            "LOOKAHEAD": LOOKAHEAD,
                            "nprows": nprows,
                            "NSUP": NSUP,
                            "NREL": NREL
                        }
                        func_eval = search_func_eval(function_evaluations, point)
                        if func_eval is not None:
                            x_names = {'1':"MMD_ATA", '2':"MMD_AT_PLUS_A", '3':"COLAMD", '4':"METIS_AT_PLUS_A", '5':"PARMETIS"}
                            x_observed.append(x_names[COLPERM])
                            #y_observed.append(func_eval["evaluation_result"]["time"])
                            #y_observed.append(np.average(func_eval["evaluation_detail"]["time"]["evaluations"]))
                            result_arr = func_eval["evaluation_detail"]["time"]["evaluations"]
                            result_arr.remove(min(result_arr))
                            result_arr.remove(max(result_arr))
                            y_observed.append(np.average(result_arr))
                        #x_labels_arr.append(str(LOOKAHEAD)+"-"+str(nprows)+"-"+str(NSUP)+"-"+str(NREL))
                elif plot_cnt == 1:
                    for LOOKAHEAD in range(5,31,1):
                        point = {
                            "matrix": matrix,
                            "COLPERM": COLPERM,
                            "LOOKAHEAD": LOOKAHEAD,
                            "nprows": nprows,
                            "NSUP": NSUP,
                            "NREL": NREL
                        }
                        func_eval = search_func_eval(function_evaluations, point)
                        if func_eval is not None:
                            x_observed.append(LOOKAHEAD)
                            #y_observed.append(func_eval["evaluation_result"]["time"])
                            #y_observed.append(np.average(func_eval["evaluation_detail"]["time"]["evaluations"]))
                            result_arr = func_eval["evaluation_detail"]["time"]["evaluations"]
                            result_arr.remove(min(result_arr))
                            result_arr.remove(max(result_arr))
                            y_observed.append(np.average(result_arr))
                        #x_labels_arr.append(str(LOOKAHEAD)+"-"+str(nprows)+"-"+str(NSUP)+"-"+str(NREL))
                elif plot_cnt == 2:
                    #for NSUP in range(30, 300, 1):
                    #for LOOKAHEAD in range(5,20,1):
                    for nprows in range(1, 257, 1):
                    #for NSUP in range(30, 300, 1):
                    #for NREL in range(10, 40, 1):
                        point = {
                            "matrix": matrix,
                            "COLPERM": COLPERM,
                            "LOOKAHEAD": LOOKAHEAD,
                            "nprows": nprows,
                            "NSUP": NSUP,
                            "NREL": NREL
                        }
                        func_eval = search_func_eval(function_evaluations, point)
                        if func_eval is not None:
                            x_observed.append(nprows)
                            #y_observed.append(func_eval["evaluation_result"]["time"])
                            #y_observed.append(np.average(func_eval["evaluation_detail"]["time"]["evaluations"]))
                            result_arr = func_eval["evaluation_detail"]["time"]["evaluations"]
                            result_arr.remove(min(result_arr))
                            result_arr.remove(max(result_arr))
                            y_observed.append(np.average(result_arr))
                        #x_labels_arr.append(str(LOOKAHEAD)+"-"+str(nprows)+"-"+str(NSUP)+"-"+str(NREL))
                elif plot_cnt == 3:
                    #for NSUP in range(30, 300, 1):
                    #for LOOKAHEAD in range(5,20,1):
                    for NSUP in range(30, 320, 1):
                    #for NREL in range(10, 40, 1):
                        point = {
                            "matrix": matrix,
                            "COLPERM": COLPERM,
                            "LOOKAHEAD": LOOKAHEAD,
                            "nprows": nprows,
                            "NSUP": NSUP,
                            "NREL": NREL
                        }
                        func_eval = search_func_eval(function_evaluations, point)
                        if func_eval is not None:
                            x_observed.append(NSUP)
                            #y_observed.append(func_eval["evaluation_result"]["time"])
                            #y_observed.append(np.average(func_eval["evaluation_detail"]["time"]["evaluations"]))
                            result_arr = func_eval["evaluation_detail"]["time"]["evaluations"]
                            result_arr.remove(min(result_arr))
                            result_arr.remove(max(result_arr))
                            y_observed.append(np.average(result_arr))
                        #x_labels_arr.append(str(LOOKAHEAD)+"-"+str(nprows)+"-"+str(NSUP)+"-"+str(NREL))
                elif plot_cnt == 4:
                    #for NSUP in range(30, 300, 1):
                    #for LOOKAHEAD in range(5,20,1):
                    #for NSUP in range(30, 300, 1):
                    for NREL in range(10, 51, 1):
                        point = {
                            "matrix": matrix,
                            "COLPERM": COLPERM,
                            "LOOKAHEAD": LOOKAHEAD,
                            "nprows": nprows,
                            "NSUP": NSUP,
                            "NREL": NREL
                        }
                        func_eval = search_func_eval(function_evaluations, point)
                        if func_eval is not None:
                            x_observed.append(NREL)
                            #y_observed.append(func_eval["evaluation_result"]["time"])
                            #y_observed.append(np.average(func_eval["evaluation_detail"]["time"]["evaluations"]))
                            result_arr = func_eval["evaluation_detail"]["time"]["evaluations"]
                            result_arr.remove(min(result_arr))
                            result_arr.remove(max(result_arr))
                            y_observed.append(np.average(result_arr))
                        #x_labels_arr.append(str(LOOKAHEAD)+"-"+str(nprows)+"-"+str(NSUP)+"-"+str(NREL))

                print ("OBSERVED SAMPLES")
                print (x_observed)
                print (y_observed)

                if plot_cnt == 0:
                    ax = axs[0][0]
                elif plot_cnt == 1:
                    ax = axs[0][1]
                elif plot_cnt == 2:
                    ax = axs[1][0]
                elif plot_cnt == 3:
                    ax = axs[1][1]
                elif plot_cnt == 4:
                    ax = axs[2][0]

                if plot_cnt == 0:
                    ax.bar(x_observed, y_observed) #, 'k', lw=3, zorder=9)
                else:
                    ax.plot(x_observed, y_observed, 'k', lw=3, zorder=9)

                if plot_cnt == 0:
                    ax.set_title("Varying COLPERM", fontsize=16)
                    #ax.set_title("Varying COLPERM, default parameters for LOOKAHEAD, nprows, NSUP, NREL", fontsize=16)
                    #ax.set_title("Varying COLPERM;  LOOKAHEAD=10, nprows=4, NSUP=128, NREL=20", fontsize=16)
                    #ax.set_xlabel("COLPERM", fontsize=16)
                    #ax.set_xticks(rotation=90)
                    #ax.tick_params(axis='x', rotation=90)
                    #ax.set_xticklabels(["\n"*(i%2) + l for i,l in enumerate(x)])
                    ax.set_xticklabels(["'MMD_ATA'", "\n'MMD_AT_PLUS_A'", "'COLAMD'", "\n'METIS_AT_PLUS_A'", "'PARMETIS'"])
                    ax.set_yticks([0,5,10])
                    ax.set_yticklabels([0,5,10], fontsize=16)
                elif plot_cnt == 1:
                    ax.set_title("Varying LOOKAHEAD", fontsize=16)
                    #ax.set_title("Varying LOOKAHEAD; COLPERM='METIS_AT_PLUS_A', nprows=4, NSUP=128, NREL=20", fontsize=16)
                    #ax.set_xlabel("LOOKAHEAD", fontsize=16)
                    ax.set_xticks([5,10,15,20,25,30])
                    ax.set_xticklabels([5,10,15,20,25,30], fontsize=16)
                    ax.set_yticks([1.5,2,2.5])
                    ax.set_yticklabels([1.5,2,2.5], fontsize=16)
                elif plot_cnt == 2:
                    ax.set_title("Varying nprows", fontsize=16)
                    #ax.set_title("Varying nprows; COLPERM='METIS_AT_PLUS_A', LOOKAHEAD=10, NSUP=128, NREL=20", fontsize=16)
                    #ax.set_xlabel("nprows", fontsize=16)
                    ax.set_xticks([1,2,4,6,8,10,11])
                    ax.set_xticklabels([1,2,4,6,8,10,11], fontsize=16)
                    ax.set_yticks([1,2,3,4])
                    ax.set_yticklabels([1,2,3,4], fontsize=16)
                elif plot_cnt == 3:
                    ax.set_title("Varying NSUP", fontsize=16)
                    #ax.set_title("Varying NSUP; COLPERM='METIS_AT_PLUS_A', LOOKAHEAD=10, nprows=4, NREL=20", fontsize=16)
                    #ax.set_xlabel("NSUP", fontsize=16)
                    ax.set_xticks([50,100,150,200,250,300])
                    ax.set_xticklabels([50,100,150,200,250,300], fontsize=16)
                    ax.set_yticks([1.5,2,2.5])
                    ax.set_yticklabels([1.5,2,2.5], fontsize=16)
                elif plot_cnt == 4:
                    #ax.set_title("Varying NREL; COLPERM='METIS_AT_PLUS_A', LOOKAHEAD=10, nprows=4, NSUP=128", fontsize=16)
                    ax.set_title("Varying NREL", fontsize=16)
                    #ax.set_xlabel("NREL", fontsize=16)
                    ax.set_xticks([10,15,20,25,30,35,40,45,50])
                    ax.set_xticklabels([10,15,20,25,30,35,40,45,50], fontsize=16)
                    ax.set_yticks([1.5,2,2.5])
                    ax.set_yticklabels([1.5,2,2.5], fontsize=16)
                #ax.set_ylabel("Measured runtime (s)", fontsize=16)

        fig.supylabel("Measured runtime (s)", fontsize=16)
        fig.suptitle("SuperLU_DIST performance for varying one paramater value (default parameter values: \nCOLPERM='METIS_AT_PLUT_A' LOOKAHED=10, nprows=4, NSUP=128, \n NREL=20) (matrix=" + matrix + ", 4 Haswell nodes, 128 MPI processes)", fontsize=16)
        fig.tight_layout()
        fig.savefig("manual_analysis_"+str(matrix)+".pdf")

