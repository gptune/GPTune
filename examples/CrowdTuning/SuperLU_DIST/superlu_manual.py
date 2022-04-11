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
mpirun -n 8 python superlu_TLA_base.py -nprocmin_pernode 1 -ntask 20 -nrun 800 -obj time

where:
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -obj is the tuning objective: "time" or "memory"
"""

import sys
import os
import numpy as np
import argparse

import mpi4py
from mpi4py import MPI
from array import array
import math

#sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
#
#from gptune import *
#from autotune.problem import *
#from autotune.space import *
#from autotune.search import *
import time
import json

def objectives(point): # should always use this name for user-defined objective function

    print ("POINT: ", point)

    nodes = point['nodes']
    cores = point['cores']
    target = point['target']

    matrix = point['matrix']
    COLPERM = point['COLPERM']
    LOOKAHEAD = point['LOOKAHEAD']
    nprows = point['nprows']

    npernode = point['npernode']
    nproc = nodes*npernode
    nthreads = int(cores / npernode)

    NSUP = point['NSUP']
    NREL = point['NREL']
    npcols = int(nproc / nprows)
    params = [matrix, 'COLPERM', COLPERM, 'LOOKAHEAD', LOOKAHEAD, 'nthreads', nthreads, 'npernode', npernode, 'nprows', nprows, 'npcols', npcols, 'NSUP', NSUP, 'NREL', NREL]
    RUNDIR = os.path.abspath(__file__ + "/../superlu_dist/build/EXAMPLE")
    INPUTDIR = os.path.abspath(__file__ + "/../superlu_dist/EXAMPLE/")
    nproc = int(nprows * npcols)

    """ pass some parameters through environment variables """
    info = MPI.Info.Create()
    envstr = 'OMP_NUM_THREADS=%d\n' %(nthreads)
    envstr += 'NREL=%d\n' %(NREL)
    envstr += 'NSUP=%d\n' %(NSUP)
    info.Set('env',envstr)
    info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works

    """ use MPI spawn to call the executable, and pass the other parameters and inputs through command line """
    print('exec', "%s/pddrive_spawn"%(RUNDIR), 'args', ['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], 'nproc', nproc, 'env', 'OMP_NUM_THREADS=%d' %(nthreads), 'NSUP=%d' %(NSUP), 'NREL=%d' %(NREL))
    comm = MPI.COMM_SELF.Spawn("%s/pddrive_spawn"%(RUNDIR), args=['-c', '%s'%(npcols), '-r', '%s'%(nprows), '-l', '%s'%(LOOKAHEAD), '-p', '%s'%(COLPERM), '%s/%s'%(INPUTDIR,matrix)], maxprocs=nproc,info=info)

    """ gather the return value using the inter-communicator, also refer to the INPUTDIR/pddrive_spawn.c to see how the return value are communicated """
    tmpdata = array('f', [0,0])
    comm.Reduce(sendbuf=None, recvbuf=[tmpdata,MPI.FLOAT],op=MPI.MAX,root=mpi4py.MPI.ROOT)
    comm.Disconnect()
    time.sleep(5.0)

    if(target=='time'):
        retval = tmpdata[0]
        print(params, ' superlu time: ', retval)

    if(target=='memory'):
        retval = tmpdata[1]
        print(params, ' superlu memory: ', retval)

    return retval

def cst1(NSUP,NREL):
    return NSUP >= NREL

def cst2(npernode,nprows,nodes):
    return nodes * npernode >= nprows

def run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL):

    json_data_arr = []
    if os.path.exists(logfile):
        with open(logfile, "r") as f_in:
            json_data_arr = json.load(f_in)
    
    skip_eval = False
    for json_data in json_data_arr:
        if matrix == json_data["task_parameter"]["matrix"] and \
           COLPERM == json_data["tuning_parameter"]["COLPERM"] and \
           LOOKAHEAD == json_data["tuning_parameter"]["LOOKAHEAD"] and \
           nprows == json_data["tuning_parameter"]["nprows"] and \
           NSUP == json_data["tuning_parameter"]["NSUP"] and \
           NREL == json_data["tuning_parameter"]["NREL"]:
            skip_eval = True
    if skip_eval == True:
        return
    
    # constants
    nodes = 4
    cores = 32
    target = "time"
    #COLPERM = '4'
    npernode = cores
    
    # cst1
    if NSUP < NREL:
        return

    # cst2
    if nodes * npernode < nprows:
        return

    results = []
    for i in range(5):
        result = objectives({
                    "matrix": matrix,
                    "LOOKAHEAD": LOOKAHEAD,
                    "nprows": nprows,
                    "NSUP": NSUP,
                    "NREL": NREL,
                    "COLPERM": COLPERM,
                    "nodes": nodes,
                    "cores": cores,
                    "target": target,
                    "npernode": npernode
                })
        result = round(result, 6)
        results.append(result)

    point = {
        "task_parameter": {
            "matrix": matrix
        },
        "machine_configuration": {
            "machine_name": "Cori",
            "haswell": {
                "nodes": nodes,
                "cores": 32
            }
        },
        "software_configuration": {
            "openmpi": {
                "version_split": [4,0,1]
            },
            "parmetis": {
                "version_split": [4,0,3]
            },
            "superlu_dist": {
                "version_split": [6,4,0]
            }
        },
        "constants": {
            "nodes": nodes,
            "cores": cores,
            "target": target,
            "npernode": npernode
        },
        "tuning_parameter": {
            "COLPERM": COLPERM,
            "LOOKAHEAD": LOOKAHEAD,
            "nprows": nprows,
            "NSUP": NSUP,
            "NREL": NREL
        },
        "evaluation_result": {
            "time": round(np.average(results),6)
        },
        "evaluation_detail": {
            "time": {
                "evaluations": results,
                "objective_scheme": "average"
            }
        },
        "source": "measure",
        "tuning": "manual_search"
    }

    print (point)

    json_data_arr.append(point)
    with open(logfile, "w") as f_out:
        json.dump(json_data_arr, f_out, indent=2)

    return

def main():

    logfile = "SuperLU_DIST-pddrive_spawn-manual_search.json"

    # task parameter
    #for matrix in ["Si5H12.mtx", "Si2.mtx", "SiH4.mtx", "SiNa.mtx", "benzene.mtx"]:
    #for matrix in ["Si5H12.mtx"]: #, "Si2.mtx", "SiH4.mtx", "SiNa.mtx", "benzene.mtx"]:
    for matrix in ["Si5H12.mtx", "Si10H16.mtx", "SiO.mtx", "H2O.mtx", "GaAsH6.mtx", "Ga3As3H12.mtx"]:
        COLPERM = '4'
        LOOKAHEAD = 10
        nprows = 8
        NSUP = 128
        NREL = 20

        for COLPERM in ['1','2','3','4','5']:
            run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL)

        COLPERM = '4'
        LOOKAHEAD = 10
        nprows = 8
        NSUP = 128
        NREL = 20

        for LOOKAHEAD in range(5, 31, 1):
            run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL)

        COLPERM = '4'
        LOOKAHEAD = 10
        nprows = 8
        NSUP = 128
        NREL = 20

        for nprows in range(1, 12, 1):
            run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL)

        COLPERM = '4'
        LOOKAHEAD = 10
        nprows = 8
        NSUP = 128
        NREL = 20

        for NSUP in range(30, 321, 1):
            run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL)

        COLPERM = '4'
        LOOKAHEAD = 10
        nprows = 8
        NSUP = 128
        NREL = 20

        for NREL in range(10, 51, 1):
            run_eval(logfile, matrix, COLPERM, LOOKAHEAD, nprows, NSUP, NREL)

    return

if __name__ == "__main__":

    main()
