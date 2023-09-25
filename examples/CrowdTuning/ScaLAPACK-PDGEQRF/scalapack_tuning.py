#! /usr/bin/env python3
# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#
################################################################################

"""
Example of invocation of this script:

python scalapack_tuning.py -input_m 5000 -input_n 5000 -nprocmin_pernode 1 -nrun 10 -jobid 0 -bunit 8

where:
    -input_m (input_n) is the number of rows (columns) of the target matrix
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -nrun is the number of calls per task 
    -jobid is optional. You can always set it to 0.
"""

################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))
api_key = os.getenv("CROWDTUNING_API_KEY")

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import numpy as np
import argparse
import pickle
from random import *
# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import time
import math

import crowdtune

################################################################################

''' The objective function required by GPTune. '''
# should always use this name for user-defined objective function
def objectives(point):                          
	print('objective is not needed when options["RCI_mode"]=True')

def cst1(mb,p,m,bunit):
    return mb*bunit * p <= m
def cst2(nb,lg2npernode,n,p,nodes,bunit):
    return nb * bunit * nodes * 2**lg2npernode <= n * p
def cst3(lg2npernode,p,nodes):
    return nodes * 2**lg2npernode >= p

def LoadSourceFunctionEvaluations(nodes, cores, nprocmin_pernode):

    source_function_evaluations = []

    for tid in range(1):
        if tid == 0:
            m_src = 10000
            n_src = 10000
        elif tid == 1:
            m_src = 8000
            n_src = 8000
        elif tid == 2:
            m_src = 6000
            n_src = 6000

        problem_space = {
            "input_space": [
                {"name":"m", "value": m_src},
                {"name":"n", "value": n_src}
            ],
            "constants": [
                {"nodes":nodes, "cores":cores, "bunit":8}
            ],
            "parameter_space": [
                {"name":"mb", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":16},
                {"name":"nb", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":16},
                {"name":"lg2npernode", "type":"integer", "transformer":"normalize", "lower_bound":int(math.log2(nprocmin_pernode)), "upper_bound":int(math.log2(cores))},
                {"name":"p", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":nodes*cores},
            ],
            "output_space": [
                {"name":"r", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("Inf")}
            ]
        }

        configuration_space = {}

        ret = crowdtune.QueryFunctionEvaluations(api_key = api_key,
                tuning_problem_name = "PDGEQRF",
                problem_space = problem_space,
                configuration_space = configuration_space)

        print ("crowdtuning API, number of downloaded function evaluations: ", len(ret))

        source_function_evaluations.append(ret)

    return source_function_evaluations

def main():

    # Parse command line arguments
    args = parse_args()
    input_m = args.input_m
    input_n = args.input_n
    bunit = args.bunit
    nprocmin_pernode = args.nprocmin_pernode
    nrun = args.nrun
    npilot = args.npilot
    JOBID = args.jobid
    nbatch = args.nbatch
    tuning_method = args.tuning_method

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.system("mkdir -p scalapack-driver/bin/%s;" %(machine))
    DRIVERFOUND=False
    INSTALLDIR=os.getenv('GPTUNE_INSTALL_PATH')
    DRIVER = os.path.abspath(__file__ + "/../../../build/pdqrdriver")
    if(os.path.exists(DRIVER)):
        DRIVERFOUND=True
    elif(INSTALLDIR is not None):
        DRIVER = INSTALLDIR+"/gptune/pdqrdriver"
        if(os.path.exists(DRIVER)):
            DRIVERFOUND=True
    else:
        for p in sys.path:
            if("gptune" in p):
                DRIVER=p+"/pdqrdriver"
                if(os.path.exists(DRIVER)):
                    DRIVERFOUND=True
                    break

    if(DRIVERFOUND == True):
        os.system("cp %s scalapack-driver/bin/%s/.;" %(DRIVER,machine))
    else:
        raise Exception(f"pdqrdriver cannot be located. Try to set env variable GPTUNE_INSTALL_PATH correctly.")

    nprocmax = nodes*cores

    mmin=128
    nmin=128

    m = Integer(mmin, input_m, transform="normalize", name="m")
    n = Integer(nmin, input_n, transform="normalize", name="n")
    mb = Integer(1, 16, transform="normalize", name="mb")
    nb = Integer(1, 16, transform="normalize", name="nb")
    lg2npernode = Integer(int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="lg2npernode")
    p = Integer(1, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([mb, nb, lg2npernode, p])
    OS = Space([r])
    
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    constants={"nodes":nodes,"cores":cores,"bunit":bunit}
    # print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    data = Data(problem)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_random_seed'] = nbatch
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = nbatch
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = nbatch
    options['verbose'] = False
    options['RCI_mode'] = True
    #options.validate(computer=computer)

    if tuning_method == "SLA":
        options["TLA_method"] = None
        options.validate(computer=computer)
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
        giventask = [input_m, input_n]

        (data, modeler, stats) = gt.SLA(NS=nrun, NS1=npilot, Tgiven=giventask)

        """ Print all input and parameter samples """
        print("    m:%d n:%d" % (data.I[0], data.I[1]))
        print("    Ps ", data.P)
        print("    Os ", data.O.tolist())
        print('    Popt ', data.P[np.argmin(data.O)], 'Oopt ', min(data.O)[0], 'nth ', np.argmin(data.O))

    else:
        if tuning_method == "TLA_Sum":
            options["TLA_method"] = "Sum"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Regression":
            options["TLA_method"] = "Regression"
            options.validate(computer=computer)
        elif tuning_method == "TLA_LCM_BF":
            options["TLA_method"] = "LCM_BF"
            options.validate(computer=computer)
        elif tuning_method == "TLA_LCM":
            options["TLA_method"] = "LCM"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Stacking":
            options["TLA_method"] = "Stacking"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Toggling":
            options["TLA_method"] = "Ensemble_Toggling"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Peeking":
            options["TLA_method"] = "Ensemble_Peeking"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_Prob":
            options["TLA_method"] = "Ensemble_Prob"
            options.validate(computer=computer)
        elif tuning_method == "TLA_Ensemble_ProbDyn":
            options["TLA_method"] = "Ensemble_ProbDyn"
            options.validate(computer=computer)
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))
        giventask = [[input_m, input_n]]
        (data, modeler, stats) = gt.TLA_I(NS=nrun, Tnew=giventask, source_function_evaluations=LoadSourceFunctionEvaluations(nodes, cores, nprocmin_pernode))

        tid = 0
        print("tid: %d" % (tid))
        print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-input_m', type=int, default=-1, help='Number of rows')
    parser.add_argument('-input_n', type=int, default=-1, help='Number of columns')    
    parser.add_argument('-bunit', type=int, default=8, help='mb and nb are integer multiples of bunit')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments    
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, help='Number of pilot samples per task')
    
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')

    parser.add_argument('-nbatch', type=int, default=0, help='Input task t value')
    parser.add_argument('-tuning_method', type=str, default='SLA', help='Tuning method')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
