#! /usr/bin/env python3

import os

config = "cori-haswell-openmpi-gnu-1nodes"

def merge_db(db_files_list):

    import json

    merged_json_data = {}

    merged_json_data["tuning_problem_name"] = "PDGEQRF"
    merged_json_data["surrogate_model"] = []
    merged_json_data["func_eval"] = []

    nargs = len(db_files_list)

    print ("nargs: ", nargs)

    for i in range(1, nargs, 1):

        with open(db_files_list[i], "r") as f_in:
            json_data = json.load(f_in)

            for surrogate_model in json_data["surrogate_model"]:
                merged_json_data["surrogate_model"].append(surrogate_model)
            for func_eval in json_data["func_eval"]:
                merged_json_data["func_eval"].append(func_eval)

    with open("db.out", "w") as f_out:

        json.dump(merged_json_data, f_out, indent=2)

    return

def run_tla5():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for task in [6000,8000,10000,12000,14000]:
        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
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

            os.system("rm -rf gptune.db")
            os.system("rm -rf a.out.log")

            os.system("rm -rf db.out")

            db_files_list = []
            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task1)+"-50/PDGEQRF.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task2)+"-50/PDGEQRF.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task3)+"-50/PDGEQRF.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task4)+"-50/PDGEQRF.json")
            merge_db(db_files_list)

            os.system("mkdir -p gptune.db")
            os.system("mv db.out gptune.db/PDGEQRF.json")

            dest_dir = "TLA_experiments/TLA5_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)
            if os.path.exists(dest_dir+"/a.out.log"):
                print (dest_dir + " data already exists!")
                continue
            else:
                os.system("rm -rf "+dest_dir)

                os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(max(task,transfer_task1,transfer_task2,transfer_task3,transfer_task4))+" "+str(5)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task1)+" "+str(transfer_task2)+" "+str(transfer_task3)+" "+str(transfer_task4))

                os.system("mkdir -p "+dest_dir)
                os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla4():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for task in [6000,8000,10000,12000,14000]:
        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
            for transfer_task1 in [6000,8000,10000,12000,14000]:
                for transfer_task2 in [6000,8000,10000,12000,14000]:
                    for transfer_task3 in [6000,8000,10000,12000,14000]:
                        if task != transfer_task1 and task != transfer_task2 and task != transfer_task3 and\
                           transfer_task1 != transfer_task2 and transfer_task1 != transfer_task3 and\
                           transfer_task2 != transfer_task3:
                            os.system("rm -rf gptune.db")
                            os.system("rm -rf a.out.log")

                            os.system("rm -rf db.out")

                            db_files_list = []
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task1)+"-50/PDGEQRF.json")
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task2)+"-50/PDGEQRF.json")
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task3)+"-50/PDGEQRF.json")
                            merge_db(db_files_list)

                            os.system("mkdir -p gptune.db")
                            os.system("mv db.out gptune.db/PDGEQRF.json")

                            dest_dir = "TLA_experiments/TLA4_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)
                            if os.path.exists(dest_dir+"/a.out.log"):
                                print (dest_dir + " data already exists!")
                                continue
                            else:
                                os.system("rm -rf "+dest_dir)

                                os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(max(task,transfer_task1,transfer_task2,transfer_task3))+" "+str(4)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task1)+" "+str(transfer_task2)+" "+str(transfer_task3))

                                os.system("mkdir -p "+dest_dir)
                                os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                                os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla3():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for task in [6000,8000,10000,12000,14000]:
        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
            for transfer_task1 in [6000,8000,10000,12000,14000]:
                for transfer_task2 in [6000,8000,10000,12000,14000]:
                    if task != transfer_task1 and task != transfer_task2 and transfer_task1 != transfer_task2:
                        os.system("rm -rf gptune.db")
                        os.system("rm -rf a.out.log")

                        os.system("rm -rf db.out")

                        db_files_list = []
                        db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task1)+"-50/PDGEQRF.json")
                        db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task2)+"-50/PDGEQRF.json")
                        db_files_list.append("TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task3)+"-50/PDGEQRF.json")
                        merge_db(db_files_list)

                        os.system("mkdir -p gptune.db")
                        os.system("mv db.out gptune.db/PDGEQRF.json")

                        dest_dir = "TLA_experiments/TLA3_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)
                        if os.path.exists(dest_dir+"/a.out.log"):
                            print (dest_dir + " data already exists!")
                            continue
                        else:
                            os.system("rm -rf "+dest_dir)

                            os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(max(task,transfer_task1,transfer_task2))+" "+str(3)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task1)+" "+str(transfer_task2))

                            os.system("mkdir -p "+dest_dir)
                            os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                            os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla2():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for task in [6000,8000,10000,12000,14000]:
        for nrun in [10,20,30,40,50]:
            for transfer_task in [6000,8000,10000,12000,14000]:
                if task != transfer_task:
                    os.system("rm -rf gptune.db")
                    os.system("rm -rf a.out.log")

                    os.system("cp -r TLA_experiments/SLA-GPTune-"+config+"-"+str(transfer_task)+"-50 gptune.db")

                    dest_dir = "TLA_experiments/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)
                    if os.path.exists(dest_dir+"/a.out.log"):
                        print (dest_dir + " data already exists!")
                        continue
                    else:
                        os.system("rm -rf "+dest_dir)

                        os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(max(task,transfer_task))+" "+str(2)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task))

                        os.system("mkdir -p "+dest_dir)
                        os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                        os.system("mv a.out.log "+dest_dir+"/a.out.log")

    for task in [6000,8000,10000,12000,14000]:
        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
            config = "cori-haswell-openmpi-gnu-1nodes"
            transfer_config = "cori-knl-openmpi-gnu-1nodes"

            os.system("rm -rf gptune.db")
            os.system("rm -rf a.out.log")

            os.system("cp -r TLA_experiments/SLA-GPTune-"+transfer_config+"-"+str(task)+"-50 gptune.db")

            dest_dir = "TLA_experiments/TLA2_machine-"+config+"-"+transfer_config+"-"+str(task)+"-"+str(nrun)
            if os.path.exists(dest_dir+"/a.out.log"):
                print (dest_dir + " data already exists!")
                continue
            else:
                os.system("rm -rf "+dest_dir)

                os.system("./run_cori_scalapack.sh TLA_machine "+config+" "+str(task)+" "+str(2)+" "+str(nrun)+" "+str(1)+" "+"GPTune")

                os.system("mkdir -p "+dest_dir)
                os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_sla():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for optimization in ["GPTune","opentuner","hpbandster"]:
        for task in [6000,8000,10000,12000,14000]: # Task value
            for nrun in [10,20,30,40,50]:
                if optimization == "opentuner":
                    os.system("rm -rf opentuner.db")
                    os.system("rm -rf opentuner.log")

                os.system("rm -rf gptune.db")
                os.system("rm -rf a.out.log")

                dest_dir = "TLA_experiments/SLA-"+optimization+"-"+config+"-"+str(task)+"-"+str(nrun)
                if os.path.exists(dest_dir+"/a.out.log"):
                    print (dest_dir + " data already exists!")
                    continue
                else:
                    os.system("rm -rf "+dest_dir)
                    os.system("mkdir -p "+dest_dir)

                    os.system("./run_cori_scalapack.sh MLA "+config+" "+str(task)+" "+str(1)+" "+str(nrun)+" "+str(1)+" "+optimization)

                    if optimization == "GPTune":
                        os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                    os.system("mv a.out.log "+dest_dir+"/a.out.log")

def run_sla_long():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for optimization in ["GPTune"]:
        for task in [10000,6000,8000,12000,14000]: # Task value
            for nrun in [100,200]:
                if optimization == "opentuner":
                    os.system("rm -rf opentuner.db")
                    os.system("rm -rf opentuner.log")

                os.system("rm -rf gptune.db")
                os.system("rm -rf a.out.log")

                dest_dir = "TLA_experiments/SLA-"+optimization+"-"+config+"-"+str(task)+"-"+str(nrun)
                if os.path.exists(dest_dir+"/a.out.log"):
                    print (dest_dir + " data already exists!")
                    continue
                else:
                    os.system("rm -rf "+dest_dir)
                    os.system("mkdir -p "+dest_dir)

                    os.system("./run_cori_scalapack.sh MLA "+config+" "+str(task)+" "+str(1)+" "+str(nrun)+" "+str(1)+" "+optimization)

                    if optimization == "GPTune":
                        os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                    os.system("mv a.out.log "+dest_dir+"/a.out.log")

if __name__ == "__main__":

    #run_sla()
    #run_tla2()
    #run_tla3()
    #run_tla4()
    #run_tla5()

    run_sla_long()
