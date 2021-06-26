#! /usr/bin/env python

import sys
import os
import json
import re

number_re = "+([0-9]+\.[01-9]+|[0-9]+)"

def main():

    with open("tla2_details.csv", "w") as f_out:
        f_out.write("task-nrun, SLA, TLA (near), TLA (far)\n")

        ########

        for nrun in [10,20,30,40,50]:
            for task in [6000,8000,10000,12000,14000]:
                f_out.write(str(task)+"-"+str(nrun))

                config = "cori-haswell-openmpi-gnu-1nodes"

                with open("TLA_experiments/SLA-GPTune-"+config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                    json_data = json.load(f_in)
                    nth = 0
                    oopt = 1000000.0
                    func_eval = json_data["func_eval"]
                    for i in range(len(func_eval)):
                        if func_eval[i]["evaluation_result"]["r"] < oopt:
                            oopt = func_eval[i]["evaluation_result"]["r"]
                            nth = i
                    f_out.write(","+str(oopt))

                #for transfer_task in [0.6,0.8,1.0,1.2,1.4]:
                #    if task != transfer_task:
                #        if os.path.exists("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json"):
                #            with open("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                #                json_data = json.load(f_in)
                #                oopt = 1000000.0
                #                for func_eval in json_data["func_eval"]:
                #                    if func_eval["task_parameter"]["m"] == task:
                #                        if func_eval["evaluation_result"]["r"] < oopt:
                #                            oopt = func_eval["evaluation_result"]["r"]
                #                f_out.write(","+str(oopt))
                #        else:
                #            f_out.write(",")
                #    else:
                #        f_out.write(",")

                if task == 6000:
                    transfer_task = 8000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))
                    transfer_task = 14000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))
                elif task == 8000:
                    oopt = 0
                    transfer_task = 6000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                    transfer_task = 10000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                        f_out.write(","+str(oopt/2))
                    oopt = 0
                    transfer_task = 14000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))

                elif task == 10000:
                    transfer_task = 8000
                    oopt = 0
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                    transfer_task = 12000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                        f_out.write(","+str(oopt/2))

                    oopt = 0
                    transfer_task = 6000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                    transfer_task = 14000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                        f_out.write(","+str(oopt/2))

                elif task == 12000:
                    transfer_task = 10000
                    oopt = 0
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                    transfer_task = 14000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt_ = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt_:
                                    oopt_ = func_eval["evaluation_result"]["r"]
                        oopt += oopt_
                        f_out.write(","+str(oopt/2))
                    oopt = 0
                    transfer_task = 6000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))

                if task == 14000:
                    transfer_task = 12000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))
                    transfer_task = 6000
                    with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))
                f_out.write("\n")

    with open("analysis.csv", "w") as f_out:
        f_out.write("task-nrun,opentuner,hpbandster,gptune,TLA2(worst),TLA2(best),TLA3(worst),TLA3(best),TLA4(worst),TLA4(best),TLA5\n")

        ########

        for nrun in [10,20,30,40,50]:
            for task in [6000,8000,10000,12000,14000]:
                f_out.write(str(task)+"-"+str(nrun))

                config = "cori-haswell-openmpi-gnu-1nodes"

                with open("TLA_experiments/SLA-opentuner-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[8])
                    oopt = float(elems[6])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-hpbandster-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[8])
                    oopt = float(elems[6])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-GPTune-"+config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                    json_data = json.load(f_in)
                    nth = 0
                    oopt = 1000000.0
                    func_eval = json_data["func_eval"]
                    for i in range(len(func_eval)):
                        if func_eval[i]["evaluation_result"]["r"] < oopt:
                            oopt = func_eval[i]["evaluation_result"]["r"]
                            nth = i
                    f_out.write(","+str(oopt))

                #for transfer_task in [0.6,0.8,1.0,1.2,1.4]:
                #    if task != transfer_task:
                #        if os.path.exists("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json"):
                #            with open("TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                #                json_data = json.load(f_in)
                #                oopt = 1000000.0
                #                for func_eval in json_data["func_eval"]:
                #                    if func_eval["task_parameter"]["m"] == task:
                #                        if func_eval["evaluation_result"]["r"] < oopt:
                #                            oopt = func_eval["evaluation_result"]["r"]
                #                f_out.write(","+str(oopt))
                #        else:
                #            f_out.write(",")
                #    else:
                #        f_out.write(",")

                tla2_worst = "-"
                tla2_best = "-"
                for transfer_task in [6000,8000,10000,12000,14000]:
                    if task != transfer_task:
                        if os.path.exists("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json"):
                            with open("TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                                json_data = json.load(f_in)
                                oopt = 1000000.0
                                for func_eval in json_data["func_eval"]:
                                    if func_eval["task_parameter"]["m"] == task:
                                        if func_eval["evaluation_result"]["r"] < oopt:
                                            oopt = func_eval["evaluation_result"]["r"]
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
                for transfer_task1 in [6000,8000,10000,12000,14000]:
                    for transfer_task2 in [6000,8000,10000,12000,14000]:
                        if task != transfer_task1 and task != transfer_task2 and transfer_task1 != transfer_task2:
                            if os.path.exists("TLA_experiments/TLA3_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)+"/PDGEQRF.json"):
                                with open("TLA_experiments/TLA3_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                                    json_data = json.load(f_in)
                                    oopt = 1000000.0
                                    for func_eval in json_data["func_eval"]:
                                        if func_eval["task_parameter"]["m"] == task:
                                            if func_eval["evaluation_result"]["r"] < oopt:
                                                oopt = func_eval["evaluation_result"]["r"]
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
                for transfer_task1 in [6000,8000,10000,12000,14000]:
                    for transfer_task2 in [6000,8000,10000,12000,14000]:
                        for transfer_task3 in [6000,8000,10000,12000,14000]:
                            if task != transfer_task1 and task != transfer_task2 and task != transfer_task3 and\
                               transfer_task1 != transfer_task2 and transfer_task1 != transfer_task3 and\
                               transfer_task2 != transfer_task3:
                                if os.path.exists("TLA_experiments/TLA4_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)+"/PDGEQRF.json"):
                                    with open("TLA_experiments/TLA4_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                                        json_data = json.load(f_in)
                                        oopt = 1000000.0
                                        for func_eval in json_data["func_eval"]:
                                            if func_eval["task_parameter"]["m"] == task:
                                                if func_eval["evaluation_result"]["r"] < oopt:
                                                    oopt = func_eval["evaluation_result"]["r"]
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
                if task == 6000:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 8000, 10000, 12000, 14000
                elif task == 8000:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 6000, 10000, 12000, 14000
                elif task == 10000:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 6000, 8000, 12000, 14000
                elif task == 12000:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 6000, 8000, 10000, 14000
                elif task == 14000:
                    transfer_task1, transfer_task2, transfer_task3, transfer_task4 = 6000, 8000, 10000, 12000
                if os.path.exists("TLA_experiments/TLA5_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)+"/PDGEQRF.json"):
                    with open("TLA_experiments/TLA5_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if func_eval["evaluation_result"]["r"] < oopt:
                                    oopt = func_eval["evaluation_result"]["r"]
                        tla5 = oopt
                f_out.write(","+str(tla5))

                f_out.write("\n")

    with open("analysis_machine.csv", "w") as f_out:
        f_out.write("task-nrun,opentuner,hpbandster,gptune,TLA\n")

        for nrun in [20]:
            for task in [6000,8000,10000,12000,14000]:
                f_out.write(str(task)+"-"+str(nrun))

                config = "cori-haswell-openmpi-gnu-1nodes"

                with open("TLA_experiments/SLA-opentuner-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[8])
                    oopt = float(elems[6])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-hpbandster-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                    lastline = str(list(f_in)[-1])
                    elems = lastline.split()
                    nth = int(elems[8])
                    oopt = float(elems[6])
                    f_out.write(","+str(oopt))

                with open("TLA_experiments/SLA-GPTune-"+config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                    json_data = json.load(f_in)
                    nth = 0
                    oopt = 1000000.0
                    func_eval = json_data["func_eval"]
                    for i in range(len(func_eval)):
                        if func_eval[i]["evaluation_result"]["r"] < oopt:
                            oopt = func_eval[i]["evaluation_result"]["r"]
                            nth = i
                    f_out.write(","+str(oopt))

                transfer_config = "cori-knl-openmpi-gnu-1nodes"
                if os.path.exists("TLA_experiments/TLA2_machine-"+config+"-"+transfer_config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json"):
                    with open("TLA_experiments/TLA2_machine-"+config+"-"+transfer_config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
                        json_data = json.load(f_in)
                        oopt = 1000000.0
                        for func_eval in json_data["func_eval"]:
                            if func_eval["task_parameter"]["m"] == task:
                                if "haswell" in func_eval["machine_configuration"]:
                                    if func_eval["evaluation_result"]["r"] < oopt:
                                        oopt = func_eval["evaluation_result"]["r"]
                        f_out.write(","+str(oopt))
                f_out.write("\n")

if __name__ == "__main__":
    main()
