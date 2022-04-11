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

import os
api_key = os.getenv("CROWDTUNE_API_KEY")

import crowdtune

problem_space = {
    "input_space": [
        {"name":"matrix", "type":"categorical", "transformer":"onehot", "categories":["Si5H12.mtx","Si10H16.mtx","SiO.mtx"]}
    ],
    "constants": [
        {"cores":32, "nodes":4, "npernode": 32}
    ],
    "parameter_space": [
        {"name":"COLPERM", "type":"categorical", "transformer":"onehot", "categories":['1','2','3','4','5']},
        {"name":"LOOKAHEAD", "type":"integer", "transformer":"normalize", "lower_bound":5, "upper_bound":20},
        {"name":"nprows", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":11},
        {"name":"NSUP", "type":"integer", "transformer":"normalize", "lower_bound":30, "upper_bound":300},
        {"name":"NREL", "type":"integer", "transformer":"normalize", "lower_bound":10, "upper_bound":40}
    ],
    "output_space": [
        {"name":"runtime", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("inf")}
    ]
}

task="Si5H12.mtx" # "Si10H16.mtx" "SiO.mtx"
ret = crowdtune.QuerySensitivityAnalysis(
    api_key = api_key,
    tuning_problem_name = "SuperLU_DIST-pddrive_spawn",
    problem_space = problem_space,
    input_task = [task])
print ("Task: ", task)
print (ret)

