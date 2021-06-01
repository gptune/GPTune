#! /usr/bin/env python3

import os

def run_scenarios():

    config = "cori-haswell-openmpi-gnu-1nodes"

    for optimization in ["GPTune","opentuner","hpbandster","Random"]:
        for dataset in ["susy_10Kn","susy_100Kn","mnist_10Kn","mnist_500Kn"]:
            for nrun in [20,50]:
                os.system("rm .gptune/meta.json")
                os.system("cp .gptune/configs/"+config+".json .gptune/meta.json")

                if optimization == "opentuner":
                    os.system("rm -rf opentuner.db")
                    os.system("rm -rf opentuner.log")

                os.system("rm -rf gptune.db")
                os.system("rm -rf a.out.log")

                dest_dir = "Scenarios/STRUMPACK-KRR-HSS-"+optimization+"-"+config+"-"+str(dataset)+"-"+str(nrun)

                if os.path.exists(dest_dir+"/a.out.log"):
                    print (dest_dir + " data already exists!")
                    continue
                else:
                    os.system("./run_cori_experiments.sh "+str(dataset)+" "+str(nrun)+" "+optimization)

                    os.system("rm -rf "+dest_dir)
                    os.system("mkdir -p "+dest_dir)
                    if optimization == "GPTune":
                        os.system("mv gptune.db/STRUMPACK-KRR-HSS.json "+dest_dir+"/STRUMPACK-KRR-HSS.json")
                    os.system("mv a.out.log "+dest_dir+"/a.out.log")

if __name__ == "__main__":

    run_scenarios()

