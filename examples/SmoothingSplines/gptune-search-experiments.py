#! /usr/bin/env python

import os

def run_synthetic():
    os.system("rm -rf gptune-search-gpy.db")
    os.system("rm -rf gptune-search-lcm.db")
    os.system("mkdir gptune-search-gpy.db")
    os.system("mkdir gptune-search-lcm.db")

    for nrun in [20]:
        for t in [0.5,1.0,2.0,3.0,4.0,5.0]:
            for size in [10000,100000]:
                for v in [0.01, 0.05, 0.1]:

                    ## GPy

                    os.system("rm -rf gptune.db")

                    command = "mpirun -n 1 python gptune-search-gpy.py"
                    command += " -nrun 20"
                    command += " -var "+str(v)
                    command += " -task "+str(t)
                    command += " -size "+str(size)
    
                    output_path = "gptune-search-"+str(nrun)+"-"+str(t)+"-"+str(size)+"-"+str(v)

                    os.system(command + " | tee gptune-search-gpy.db/"+output_path)

                    ## LCM

                    os.system("rm -rf gptune.db")

                    command = "mpirun -n 1 python gptune-search-lcm.py"
                    command += " -nrun 20"
                    command += " -var "+str(v)
                    command += " -task "+str(t)
                    command += " -size "+str(size)
    
                    output_path = "gptune-search-"+str(nrun)+"-"+str(t)+"-"+str(size)+"-"+str(v)

                    os.system(command + " | tee gptune-search-lcm.db/"+output_path)

    return

def run_winequality():

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

    return

def run_household():

    for nrun in [20]:
        for attribute in ["Time"]:
            ## GPy

            os.system("rm -rf gptune.db")

            command = "mpirun -n 1 python gptune-search-gpy-household.py"
            command += " -nrun 20"

            output_path = "household-"+str(nrun)+"-"+str(attribute)

            os.system(command + " | tee gptune-search-gpy.db/"+output_path)

            ## LCM

            os.system("rm -rf gptune.db")

            command = "mpirun -n 1 python gptune-search-lcm-household.py"
            command += " -nrun 20"

            output_path = "household-"+str(nrun)+"-"+str(attribute)

            os.system(command + " | tee gptune-search-lcm.db/"+output_path)

    return

def main():
    #run_synthetic()
    #run_winequality()
    run_household()

if __name__ == "__main__":
    main()

