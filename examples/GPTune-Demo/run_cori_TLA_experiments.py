#! /usr/bin/env python3

import os

def merge_db(db_files_list):

    import json

    merged_json_data = {}

    merged_json_data["tuning_problem_name"] = "GPTune-Demo"
    merged_json_data["surrogate_model"] = []
    merged_json_data["func_eval"] = []

    nargs = len(db_files_list)

    print ("nargs: ", nargs)
    print ("db_files_list: ", db_files_list)

    for i in range(0, nargs, 1):

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

    optimization = "GPTune"

    for task in [0.6,0.8,1.0,1.2,1.4]:
        #for nrun in [10,20,30,40,50]:
        for nrun in [20]:
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

            os.system("rm -rf gptune.db")
            os.system("rm -rf a.out.log")

            os.system("rm -rf db.out")

            db_files_list = []
            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task1)+"-50/GPTune-Demo.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task2)+"-50/GPTune-Demo.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task3)+"-50/GPTune-Demo.json")
            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task4)+"-50/GPTune-Demo.json")
            merge_db(db_files_list)

            os.system("mkdir -p gptune.db")
            os.system("mv db.out gptune.db/GPTune-Demo.json")

            dest_dir = "TLA_experiments/TLA5_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(transfer_task4)+"-"+str(nrun)
            if os.path.exists(dest_dir+"/a.out.log"):
                print (dest_dir + " data already exists!")
                continue
            else:
                os.system("rm -rf "+dest_dir)

                run_command = "mpirun -n 1 python ./demo_TLA.py"
                run_command += " -ntask 3"
                run_command += " -nrun "+str(nrun)
                run_command += " -tvalue "+str(task)
                run_command += " -tvalue2 "+str(transfer_task1)
                run_command += " -tvalue3 "+str(transfer_task2)
                run_command += " -tvalue4 "+str(transfer_task3)
                run_command += " -tvalue5 "+str(transfer_task4)
                run_command += " -optimization "+optimization
                run_command += " | tee a.out.log"
                os.system(run_command)

                os.system("mkdir -p "+dest_dir)
                os.system("mv gptune.db/GPTune-Demo.json "+dest_dir+"/GPTune-Demo.json")
                os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla4():

    optimization = "GPTune"

    for task in [0.6,0.8,1.0,1.2,1.4]:
        for nrun in [20]:
            for transfer_task1 in [0.6,0.8,1.0,1.2,1.4]:
                for transfer_task2 in [0.6,0.8,1.0,1.2,1.4]:
                    for transfer_task3 in [0.6,0.8,1.0,1.2,1.4]:
                        if task != transfer_task1 and task != transfer_task2 and task != transfer_task3 and\
                           transfer_task1 != transfer_task2 and transfer_task1 != transfer_task3 and\
                           transfer_task2 != transfer_task3:
                            os.system("rm -rf gptune.db")
                            os.system("rm -rf a.out.log")

                            os.system("rm -rf db.out")

                            db_files_list = []
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task1)+"-50/GPTune-Demo.json")
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task2)+"-50/GPTune-Demo.json")
                            db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task3)+"-50/GPTune-Demo.json")
                            merge_db(db_files_list)

                            os.system("mkdir -p gptune.db")
                            os.system("mv db.out gptune.db/GPTune-Demo.json")

                            dest_dir = "TLA_experiments/TLA4_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(transfer_task3)+"-"+str(nrun)
                            if os.path.exists(dest_dir+"/a.out.log"):
                                print (dest_dir + " data already exists!")
                                continue
                            else:
                                os.system("rm -rf "+dest_dir)

                                run_command = "mpirun -n 1 python ./demo_TLA.py"
                                run_command += " -ntask 3"
                                run_command += " -nrun "+str(nrun)
                                run_command += " -tvalue "+str(task)
                                run_command += " -tvalue2 "+str(transfer_task1)
                                run_command += " -tvalue3 "+str(transfer_task2)
                                run_command += " -tvalue4 "+str(transfer_task3)
                                run_command += " -optimization "+optimization
                                run_command += " | tee a.out.log"
                                os.system(run_command)

                                os.system("mkdir -p "+dest_dir)
                                os.system("mv gptune.db/GPTune-Demo.json "+dest_dir+"/GPTune-Demo.json")
                                os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla3():

    optimization = "GPTune"

    for task in [0.6,0.8,1.0,1.2,1.4]:
        for nrun in [20]:
            for transfer_task1 in [0.6,0.8,1.0,1.2,1.4]:
                for transfer_task2 in [0.6,0.8,1.0,1.2,1.4]:
                    if task != transfer_task1 and task != transfer_task2 and transfer_task1 != transfer_task2:
                        os.system("rm -rf gptune.db")
                        os.system("rm -rf a.out.log")

                        os.system("rm -rf db.out")

                        db_files_list = []
                        db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task1)+"-50/GPTune-Demo.json")
                        db_files_list.append("TLA_experiments/SLA-GPTune-"+str(transfer_task2)+"-50/GPTune-Demo.json")
                        merge_db(db_files_list)

                        os.system("mkdir -p gptune.db")
                        os.system("mv db.out gptune.db/GPTune-Demo.json")

                        dest_dir = "TLA_experiments/TLA3_task-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)
                        if os.path.exists(dest_dir+"/a.out.log"):
                            print (dest_dir + " data already exists!")
                            continue
                        else:
                            os.system("rm -rf "+dest_dir)

                            run_command = "mpirun -n 1 python ./demo_TLA.py"
                            run_command += " -ntask 3"
                            run_command += " -nrun "+str(nrun)
                            run_command += " -tvalue "+str(task)
                            run_command += " -tvalue2 "+str(transfer_task1)
                            run_command += " -tvalue3 "+str(transfer_task2)
                            run_command += " -optimization "+optimization
                            run_command += " | tee a.out.log"
                            os.system(run_command)

                            os.system("mkdir -p "+dest_dir)
                            os.system("mv gptune.db/GPTune-Demo.json "+dest_dir+"/GPTune-Demo.json")
                            os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_tla2():

    optimization = "GPTune"

    for task in [0.6,0.8,1.0,1.2,1.4]:
        for nrun in [10,20,30,40,50]:
            for transfer_task in [0.6,0.8,1.0,1.2,1.4]:
                if task != transfer_task:
                    os.system("rm -rf gptune.db")
                    os.system("rm -rf a.out.log")

                    os.system("cp -r TLA_experiments/SLA-GPTune-"+str(transfer_task)+"-50 gptune.db")

                    dest_dir = "TLA_experiments/TLA2_task-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)
                    if os.path.exists(dest_dir+"/a.out.log"):
                        print (dest_dir + " data already exists!")
                        continue
                    else:
                        os.system("rm -rf "+dest_dir)

                        run_command = "mpirun -n 1 python ./demo_TLA.py"
                        run_command += " -ntask 2"
                        run_command += " -nrun "+str(nrun)
                        run_command += " -tvalue "+str(task)
                        run_command += " -tvalue2 "+str(transfer_task)
                        run_command += " -optimization "+optimization
                        run_command += " | tee a.out.log"
                        os.system(run_command)

                        os.system("mkdir -p "+dest_dir)
                        os.system("mv gptune.db/GPTune-Demo.json "+dest_dir+"/GPTune-Demo.json")
                        os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_sla():

    for optimization in ["GPTune","opentuner","hpbandster"]:
        for task in [0.6,0.8,1.0,1.2,1.4]:
            for nrun in [10,20,30,40,50]:
                if optimization == "opentuner":
                    os.system("rm -rf opentuner.db")
                    os.system("rm -rf opentuner.log")

                os.system("rm -rf gptune.db")
                os.system("rm -rf a.out.log")

                dest_dir = "TLA_experiments/SLA-"+optimization+"-"+str(task)+"-"+str(nrun)
                if os.path.exists(dest_dir+"/a.out.log"):
                    print (dest_dir + " data already exists!")
                    continue
                else:
                    os.system("rm -rf "+dest_dir)
                    os.system("mkdir -p "+dest_dir)

                    run_command = "mpirun -n 1 python ./demo.py"
                    run_command += " -ntask 1"
                    run_command += " -nrun "+str(nrun)
                    run_command += " -tvalue "+str(task)
                    run_command += " -optimization "+optimization
                    run_command += " | tee a.out.log"
                    os.system(run_command)

                    if optimization == "GPTune":
                        os.system("mv gptune.db/GPTune-Demo.json "+dest_dir+"/GPTune-Demo.json")
                    os.system("mv a.out.log "+dest_dir+"/a.out.log")

if __name__ == "__main__":

    run_sla()
    run_tla2()
    run_tla3()
    run_tla4()
    run_tla5()
