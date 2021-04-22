#! /usr/bin/env python

import os

def main():
    #os.system("rm -rf gptune-search-gpy.db")
    #os.system("rm -rf gptune-search-lcm.db")
    #os.system("mkdir gptune-search-gpy.db")
    #os.system("mkdir gptune-search-lcm.db")

    for nrun in [20]:
        for attribute in ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]:
            ## GPy

            os.system("rm -rf gptune.db")

            command = "mpirun -n 1 python gptune-search-gpy-winequality.py"
            command += " -nrun 20"
            command += " -attribute "+str(attribute)

            output_path = "winequality-"+str(nrun)+"-"+str(attribute)

            os.system(command + " | tee gptune-search-gpy.db/"+output_path)

            ## LCM

            os.system("rm -rf gptune.db")

            command = "mpirun -n 1 python gptune-search-lcm-winequality.py"
            command += " -nrun 20"
            command += " -attribute "+str(attribute)

            output_path = "winequality-"+str(nrun)+"-"+str(attribute)

            os.system(command + " | tee gptune-search-lcm.db/"+output_path)

if __name__ == "__main__":
    main()

