#! /usr/bin/env python3

import os

config = "cori-haswell-openmpi-gnu-64nodes"

def run_tla3():

    config = "cori-haswell-openmpi-gnu-64nodes"

    for task in [10000,20000,30000]:
        for nrun in [20]:
            for transfer_task1 in [10000,20000,30000]:
                for transfer_task2 in [10000,20000,30000]:
                    if task != transfer_task1 and task != transfer_task2 and trasnfer_task1 != transfer_task2:
                        os.system("rm -rf gptune.db")
                        os.system("rm -rf a.out.log")

                        os.system("rm -rf db.out")
                        merge_command = "./merge_db.py"
                        merge_command += " TLA_experiments_64nodes/SLA-GPTune-"+config+"-"+str(transfer_task1)+"-20/GPTune-Demo.json"
                        merge_command += " TLA_experiments_64nodes/SLA-GPTune-"+config+"-"+str(transfer_task2)+"-20/GPTune-Demo.json"
                        os.system(merge_command)
                        os.system("mkdir -p gptune.db")
                        os.system("mv db.out gptune.db/GPTune-Demo.json")

                        dest_dir = "TLA_experiments_64nodes/TLA3_task-"+config+"-"+str(task)+"-"+str(transfer_task1)+"-"+str(transfer_task2)+"-"+str(nrun)
                        if os.path.exists(dest_dir+"/a.out.log"):
                            print (dest_dir + " data already exists!")
                            continue
                        else:
                            os.system("rm -rf "+dest_dir)

                            os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(task)+" "+str(3)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task1)+" "+str(transfer_task2))

                            os.system("mkdir -p "+dest_dir)
                            os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                            os.system("mv a.out.log "+dest_dir+"/a.out.log")


    return

def run_tla2():

    config = "cori-haswell-openmpi-gnu-64nodes"

    for task in [10000,20000,30000]:
        for nrun in [20]:
            for transfer_task in [10000,20000,30000]:
                if task != transfer_task:
                    os.system("rm -rf gptune.db")
                    os.system("rm -rf a.out.log")

                    os.system("cp -r TLA_experiments_64nodes/SLA-GPTune-"+config+"-"+str(transfer_task)+"-20 gptune.db")

                    dest_dir = "TLA_experiments_64nodes/TLA2_task-"+config+"-"+str(task)+"-"+str(transfer_task)+"-"+str(nrun)
                    if os.path.exists(dest_dir+"/a.out.log"):
                        print (dest_dir + " data already exists!")
                        continue
                    else:
                        os.system("rm -rf "+dest_dir)
                        os.system("./run_cori_scalapack.sh TLA_task "+config+" "+str(max(task,transfer_task))+" "+str(2)+" "+str(nrun)+" "+str(1)+" "+"GPTune"+" "+str(task)+" "+str(transfer_task))

                        os.system("mkdir -p "+dest_dir)
                        os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                        os.system("mv a.out.log "+dest_dir+"/a.out.log")

    return

def run_sla():

    config = "cori-haswell-openmpi-gnu-64nodes"

    for optimization in ["GPTune","opentuner","hpbandster"]:
        for task in [10000,20000,30000]:
            for nrun in [20]:
                if optimization == "opentuner":
                    os.system("rm -rf opentuner.db")
                    os.system("rm -rf opentuner.log")

                os.system("rm -rf gptune.db")
                os.system("rm -rf a.out.log")

                dest_dir = "TLA_experiments_64nodes/SLA-"+optimization+"-"+config+"-"+str(task)+"-"+str(nrun)

                if os.path.exists(dest_dir+"/a.out.log"):
                    print (dest_dir + " data already exists!")
                    continue
                else:
                    os.system("./run_cori_scalapack.sh MLA "+config+" "+str(task)+" "+str(1)+" "+str(nrun)+" "+str(1)+" "+optimization)

                    os.system("rm -rf "+dest_dir)
                    os.system("mkdir -p "+dest_dir)
                    if optimization == "GPTune":
                        os.system("mv gptune.db/PDGEQRF.json "+dest_dir+"/PDGEQRF.json")
                    os.system("mv a.out.log "+dest_dir+"/a.out.log")

if __name__ == "__main__":

    run_sla()
    run_tla2()
    run_tla3()
    #run_tla4()
    #run_tla5()
