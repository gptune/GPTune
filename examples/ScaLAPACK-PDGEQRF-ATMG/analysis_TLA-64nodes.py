#! /usr/bin/env python

import sys
import os
import json
import re

number_re = "+([0-9]+\.[01-9]+|[0-9]+)"

def main():

    with open("analysis.csv", "w") as f_out:
        #f_out.write("task-nrun,opentuner,hpbandster,gptune,TLA2(worst),TLA2(best),TLA3(worst),TLA3(best),TLA4(worst),TLA4(best),TLA5\n")
        f_out.write("task-nrun,gptune,TLA2(worst),TLA2(best)\n")

        ########

        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
            #for task in [10000,20000,30000,40000,50000]:
            for task in [10000,20000,30000]:
                f_out.write(str(task)+"-"+str(nrun))

                config = "cori-haswell-openmpi-gnu-64nodes"

                #with open("TLA_experiments_64nodes/SLA-opentuner-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                #    lastline = str(list(f_in)[-1])
                #    elems = lastline.split()
                #    nth = int(elems[8])
                #    oopt = float(elems[6])
                #    f_out.write(","+str(oopt))

                #with open("TLA_experiments_64nodes/SLA-hpbandster-"+config+"-"+str(task)+"-"+str(nrun)+"/a.out.log", "r") as f_in:
                #    lastline = str(list(f_in)[-1])
                #    elems = lastline.split()
                #    nth = int(elems[8])
                #    oopt = float(elems[6])
                #    f_out.write(","+str(oopt))

                with open("TLA_experiments_64nodes/SLA-GPTune-"+config+"-"+str(task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
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
                #        if os.path.exists("TLA_experiments_64nodes/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json"):
                #            with open("TLA_experiments_64nodes/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
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
                #for transfer_task in [10000,20000,30000,40000,50000]:
                for transfer_task in [10000,20000,30000]:
                    if task != transfer_task:
                        if os.path.exists("TLA_experiments_64nodes/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json"):
                            with open("TLA_experiments_64nodes/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)+"/PDGEQRF.json", "r") as f_in:
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

                f_out.write("\n")

if __name__ == "__main__":
    main()
