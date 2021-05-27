#! /usr/bin/env python

import sys
import os
import json
import re

number_re = "+([0-9]+\.[01-9]+|[0-9]+)"

def main():

    with open("analysis.csv", "w") as f_out:
        f_out.write("task-nrun,opentuner,hpbandster,gptune,TLA2(worst),TLA2(best),TLA3(worst),TLA3(best),TLA4(worst),TLA4(best),TLA5\n")

        ########

        for nrun in [10,20,30,40,50]:
            for task in [0.6,0.8,1.0,1.2,1.4]:
                f_out.write(str(task)+"-"+str(nrun))

                with open("TLA_experiments/SLA-opentuner-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[5])
                    oopt = float(elems[3])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-hpbandster-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[5])
                    oopt = float(elems[3])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-GPTune-"+str(task)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                    json_data = json.load(f_in)
                    nth = 0
                    oopt = 1000000.0
                    func_eval = json_data["func_eval"]
                    for i in range(len(func_eval)):
                        if func_eval[i]["evaluation_result"]["y"] < oopt:
                            oopt = func_eval[i]["evaluation_result"]["y"]
                            nth = i
                    f_out.write(","+str(oopt))

                #for transfer_task in [0.6,0.8,1.0,1.2,1.4]:
                #    if task != transfer_task:
                #        if os.path.exists("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/GPTune-Demo.json"):
                #            with open("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                #                json_data = json.load(f_in)
                #                oopt = 1000000.0
                #                for func_eval in json_data["func_eval"]:
                #                    if func_eval["task_parameter"]["t"] == task:
                #                        if func_eval["evaluation_result"]["y"] < oopt:
                #                            oopt = func_eval["evaluation_result"]["y"]
                #                f_out.write(","+str(oopt))
                #        else:
                #            f_out.write(",")
                #    else:
                #        f_out.write(",")

                tla2_worst = "-"
                tla2_best = "-"
                for transfer_task in [0.6,0.8,1.0,1.2,1.4]:
                    if task != transfer_task:
                        if os.path.exists("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/GPTune-Demo.json"):
                            with open("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                                json_data = json.load(f_in)
                                oopt = 1000000.0
                                for func_eval in json_data["func_eval"]:
                                    if func_eval["task_parameter"]["t"] == task:
                                        if func_eval["evaluation_result"]["y"] < oopt:
                                            oopt = func_eval["evaluation_result"]["y"]
                                if tla2_worst == "-":
                                    tla2_worst = oopt
                                else:
                                    if tla2_worst < oopt:
                                        tla2_worst = oopt
                                if tla2_best == "-":
                                    tla2_best = oopt
                                else:
                                    if tla2_best > oopt:
                                        tla2_best = oopt
                f_out.write(","+str(tla2_worst)+","+str(tla2_best))

                tla3_worst = "-"
                tla3_best = "-"
                for transfer_task1 in [0.6,0.8,1.0,1.2,1.4]:
                    for transfer_task2 in [0.6,0.8,1.0,1.2,1.4]:
                        if task != transfer_task1 and task != transfer_task2 and transfer_task1 != transfer_task2:
                            if os.path.exists("TLA_experiments/TLA3_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)+"/GPTune-Demo.json"):
                                with open("TLA_experiments/TLA3_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                                    json_data = json.load(f_in)
                                    oopt = 1000000.0
                                    for func_eval in json_data["func_eval"]:
                                        if func_eval["task_parameter"]["t"] == task:
                                            if func_eval["evaluation_result"]["y"] < oopt:
                                                oopt = func_eval["evaluation_result"]["y"]
                                    if tla3_worst == "-":
                                        tla3_worst = oopt
                                    else:
                                        if tla3_worst < oopt:
                                            tla3_worst = oopt
                                    if tla3_best == "-":
                                        tla3_best = oopt
                                    else:
                                        if tla3_best > oopt:
                                            tla3_best = oopt
                f_out.write(","+str(tla3_worst)+","+str(tla3_best))

                tla4_worst = "-"
                tla4_best = "-"
                for transfer_task1 in [0.6,0.8,1.0,1.2,1.4]:
                    for transfer_task2 in [0.6,0.8,1.0,1.2,1.4]:
                        for transfer_task3 in [0.6,0.8,1.0,1.2,1.4]:
                            if task != transfer_task1 and task != transfer_task2 and task != transfer_task3 and\
                               transfer_task1 != transfer_task2 and transfer_task1 != transfer_task3 and\
                               transfer_task2 != transfer_task3:
                                if os.path.exists("TLA_experiments/TLA4_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)+"/GPTune-Demo.json"):
                                    with open("TLA_experiments/TLA4_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                                        json_data = json.load(f_in)
                                        oopt = 1000000.0
                                        for func_eval in json_data["func_eval"]:
                                            if func_eval["task_parameter"]["t"] == task:
                                                if func_eval["evaluation_result"]["y"] < oopt:
                                                    oopt = func_eval["evaluation_result"]["y"]
                                        if tla4_worst == "-":
                                            tla4_worst = oopt
                                        else:
                                            if tla4_worst < oopt:
                                                tla4_worst = oopt
                                        if tla4_best == "-":
                                            tla4_best = oopt
                                        else:
                                            if tla4_best > oopt:
                                                tla4_best = oopt
                f_out.write(","+str(tla4_worst)+","+str(tla4_best))

                tla5 = "-"
                if task == 0.6:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 0.8, 1.0, 1.2, 1.4
                elif task == 0.8:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 0.6, 1.0, 1.2, 1.4
                elif task == 1.0:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 0.6, 0.8, 1.2, 1.4
                elif task == 1.2:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 0.6, 0.8, 1.0, 1.4
                elif task == 1.4:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 0.6, 0.8, 1.0, 1.2
                if os.path.exists("TLA_experiments/TLA5_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)+"/GPTune-Demo.json"):
                    with open("TLA_experiments/TLA5_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)+"/GPTune-Demo.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["t"] == task:
                                if func_eval["evaluation_result"]["y"] < oopt:
                                    oopt = func_eval["evaluation_result"]["y"]
                        tla5 = oopt
                f_out.write(","+str(tla5))

                f_out.write("\n")

if __name__ == "__main__":
    main()
