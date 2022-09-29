#! /usr/bin/env python

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

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import *

import sys
import os
import numpy as np
import time
import argparse
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

################################################################################

def LoadFunctionEvaluations():
    api_key = os.getenv("CROWDTUNING_API_KEY")

    import crowdtune
    problem_space = {
        "input_space": [
            {"name":"mx", "value":5},
            {"name":"my", "value":7},
            {"name":"lphi", "value":1}
        ],
        "constants": [
            {"ROWPERM":'1',"COLPERM":'4',"nodes":32,"cores":32,"nstep":30}
        ],
        "parameter_space": [
            {"name":"NSUP", "type":"integer", "transformer":"normalize", "lower_bound":30, "upper_bound":300},
            {"name":"NREL", "type":"integer", "transformer":"normalize", "lower_bound":10, "upper_bound":40},
            {"name":"nbx", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":3},
            {"name":"nby", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":3},
            {"name":"npz", "type":"integer", "transformer":"normalize", "lower_bound":0, "upper_bound":5}
        ],
        "output_space": [
            {"name":"time", "type":"real", "transformer":"identity", "lower_bound":0, "upper_bound":499.9}
        ]
    }

    configuration_space = {}

    function_evaluations = crowdtune.QueryFunctionEvaluations(api_key = api_key,
        tuning_problem_name = "NIMROD_slu3d",
        problem_space = problem_space,
        configuration_space = configuration_space)
    for func_eval in function_evaluations:
        func_eval["task_parameter"]["tla_id_"] = 0

    return [function_evaluations]

# Define Problem
def objectives(point):
    print('objective is not needed when options["RCI_mode"]=True')

def main():

    args = parse_args()
    expid = args.expid
    seed = args.seed
    nstep = args.nstep
    tuning_method = args.tuning_method

    tuning_metadata = {
        "tuning_problem_name": "NIMROD_slu3d_"+tuning_method+"_"+str(expid)+"_npilot"+str(args.npilot),
        "tuning_problem_category": "NIMROD",
        "sync_crowd_repo": "no",
        "no_load_check": "yes",
        "machine_configuration": {
            "machine_name": "Cori",
            "slurm": "yes"
        },
        "software_configuration": {
            "cray-mpich": {
              "version_split": [7,7,19]
            },
            "libsci": {
              "version_split": [19,6,1]
            },
            "gcc": {
              "version_split": [8,3,0]
            },
            "parmetis": {
                "version_split": [4,0,3]
            },
            "superlu_dist": {
                "version_split": [6,4,0]
            }
        }
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    #ot.RandomGenerator.SetSeed(args.seed)
    #print(args)

    # Input parameters
    # ROWPERM   = Categoricalnorm (['1', '2'], transform="onehot", name="ROWPERM")
    # COLPERM   = Categoricalnorm (['2', '4'], transform="onehot", name="COLPERM")
    # nprows    = Integer     (0, 5, transform="normalize", name="nprows")
    # nproc    = Integer     (5, 6, transform="normalize", name="nproc")
    NSUP      = Integer     (30, 300, transform="normalize", name="NSUP")
    NREL      = Integer     (10, 40, transform="normalize", name="NREL")
    nbx      = Integer     (1, 3, transform="normalize", name="nbx")	
    nby      = Integer     (1, 3, transform="normalize", name="nby")	
    npz      = Integer     (0, 5, transform="normalize", name="npz")

    time   = Real        (float(0) , float(499.9), transform="normalize", name="time")

    # nstep      = Integer     (3, 15, transform="normalize", name="nstep")
    lphi      = Integer     (1, 3, transform="normalize", name="lphi")
    mx      = Integer     (5, 6, transform="normalize", name="mx")
    my      = Integer     (7, 8, transform="normalize", name="my")
    tla_id_ = Integer(0,1, transform="normalize", name="tla_id_")

    if tuning_method == "TLA_LCM":
        IS = Space([mx,my,lphi,tla_id_])
    else:
        IS = Space([mx,my,lphi])
    # PS = Space([ROWPERM, COLPERM, nprows, nproc, NSUP, NREL])
    # PS = Space([ROWPERM, COLPERM, NSUP, NREL, nbx, nby])
    PS = Space([NSUP, NREL, nbx, nby, npz])
    OS = Space([time])
    cst1 = "NSUP >= NREL"
    constraints = {"cst1" : cst1}
    models = {}
    constants={"ROWPERM":'1',"COLPERM":'4',"nodes":nodes,"cores":cores,"nstep":nstep}

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes = nodes, cores = cores, hosts = None)  

    """ Set and validate options """	
    options = Options()
    options['RCI_mode'] = True
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 1
    # options['objective_nprocmax'] = 1
    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16
    # options['mpi_comm'] = None
    # options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['verbose'] = False
    options['model_class'] = 'Model_GPy_LCM' # 'Model_LCM'
    options['model_random_seed'] = seed
    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_random_seed'] = seed
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = seed
    options['model_output_constraint'] = 'Ignore'

    if tuning_method == "SLA":
        options["TLA_method"] = None
    elif tuning_method == "TLA_Sum":
        options["TLA_method"] = "Sum"
    elif tuning_method == "TLA_Regression":
        options["TLA_method"] = "Regression"
    elif tuning_method == "TLA_LCM_BF":
        options["TLA_method"] = "LCM_BF"
    elif tuning_method == "TLA_LCM":
        options["TLA_method"] = "LCM"
    elif tuning_method == "TLA_Stacking":
        options["TLA_method"] = "Stacking"
    elif tuning_method == "TLA_Ensemble_Toggling":
        options["TLA_method"] = "Ensemble_Toggling"
    elif tuning_method == "TLA_Ensemble_Peeking":
        options["TLA_method"] = "Ensemble_Peeking"
    elif tuning_method == "TLA_Ensemble_Prob":
        options["TLA_method"] = "Ensemble_Prob"
    elif tuning_method == "TLA_Ensemble_ProbDyn":
        options["TLA_method"] = "Ensemble_ProbDyn"
    options.validate(computer=computer)

    data = Data(problem)
    giventask = [[5,7,1]]
    NI=len(giventask)
    
    np.set_printoptions(suppress=False, precision=4)

    NS = args.nrun
    NS1 = args.npilot

    if tuning_method == "default_parameter":
        Pdefault = [128,20,2,2,0]
        data.I = giventask
        data.P = [[Pdefault]] * NI

    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    """ Building MLA with the given list of tasks """
    #(data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=NS1)

    if tuning_method == "SLA":
        (data, modeler, stats) = gt.MLA(NS=NS, NS1=NS1, NI=1, Tgiven=giventask)
    elif tuning_method == "default_parameter":
        (data, modeler, stats) = gt.MLA(NS=1, NS1=1, NI=1, Tgiven=giventask)
    else:
        (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=LoadSourceFunctionEvaluations(tid_source=tid_source))
    # print("stats: ", stats)
    print("Sampler class: ", options['sample_class'], "Sample algo:", options['sample_algo'])
    print("Model class: ", options['model_class'])
    results_file = open(f"{args.tuning_method}_nimrod_expid{args.expid}_npilot{args.npilot}.txt", "a")
    #results_file.write(f"Tuner: {TUNER_NAME}\n")
    results_file.write(f"stats: {stats}\n")        
    if options['model_class'] == 'Model_LCM' and NI > 1:
        print("Get correlation metric ... ")
        C = model[0].M.kern.get_correlation_metric()
        print("The correlation matrix C is \n", C)
    elif options['model_class'] == 'Model_GPy_LCM' and NI > 1:
        print("Get correlation metric ... ")
        C = model[0].get_correlation_metric(NI)
        print("The correlation matrix C is \n", C)

    """ Print all input and parameter samples """
    for tid in range(NI):
        print("tid: %d"%(tid))
        print("    mx:%s my:%s lphi:%s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid])
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], f'Oopt  {min(data.O[tid])[0]:.3f}', 'nth ', np.argmin(data.O[tid]))
        results_file.write(f"tid: {tid:d}\n")
        results_file.write(f"    mx:{data.I[tid][0]:d} my:{data.I[tid][1]:d} lphi:{data.I[tid][2]:d}\n")
        # results_file.write(f"    Ps {data.P[tid]}\n")
        results_file.write(f"    Os {data.O[tid].tolist()}\n")
        # results_file.write(f'    Popt {data.P[tid][np.argmin(data.O[tid])]}  Oopt {-min(data.O[tid])[0]}  nth {np.argmin(data.O[tid])}\n')
    results_file.close()            


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bmin', type=int, default=1, help='budget min')   
    parser.add_argument('-bmax', type=int, default=1, help='budget max')   
    parser.add_argument('-eta', type=int, default=2, help='eta')
    parser.add_argument('-nstep', type=int, default=-1, help='number of time steps')   
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=1, help='total application runs')
    parser.add_argument('-npilot', type=int, default=1, help='number of pilot samples')
    parser.add_argument('-tuning_method', type=str, default='SLA', help='TLA method')
    # parser.add_argument('-sample_class', type=str,default='SampleOpenTURNS',help='Supported sample classes: SampleLHSMDU, SampleOpenTURNS')
   
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-expid', type=str, default='-', help='run id for experiment')
   
    args = parser.parse_args()
    
    return args   


if __name__ == "__main__":
    main()
