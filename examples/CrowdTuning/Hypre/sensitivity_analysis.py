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
api_key = os.getenv("CROWDTUNING_API_KEY")

import crowdtune

problem_space = {
    "input_space": [
        {"name":"nx", "type":"integer", "transformer":"normalize", "lower_bound":100, "upper_bound":101},
        {"name":"ny", "type":"integer", "transformer":"normalize", "lower_bound":100, "upper_bound":101},
        {"name":"nz", "type":"integer", "transformer":"normalize", "lower_bound":100, "upper_bound":101}
    ],
    "constants": [
        {"nodes":1, "cores":32}
    ],
    "parameter_space": [
        {"name":"Px", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":32},
        {"name":"Py", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":32},
        {"name":"Nproc", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":32},
        {"name":"strong_threshold", "type":"real", "transformer":"normalize", "lower_bound":0, "upper_bound":1},
        {"name":"trunc_factor", "type":"real", "transformer":"normalize", "lower_bound":0, "upper_bound":1},
        {"name":"P_max_elmts", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":12},
        {"name":"coarsen_type", "type":"categorical", "transformer":"onehot", "categories":['0', '1', '2', '3', '4', '6', '8', '10']},
        {"name":"relax_type", "type":"categorical", "transformer":"onehot", "categories":['-1', '0', '6', '8', '16', '18']},
        {"name":"smooth_type", "type":"categorical", "transformer":"onehot", "categories":['5', '6', '7', '8', '9']},
        {"name":"smooth_num_levels", "type":"integer", "transformer":"normalize", "lower_bound":0, "upper_bound":5},
        {"name":"interp_type", "type":"categorical", "transformer":"onehot", "categories":['0', '3', '4', '5', '6', '8', '12']},
        {"name":"agg_num_levels", "type":"integer", "transformer":"normalize", "lower_bound":0, "upper_bound":5}
    ],
    "output_space": [
        {"name":"r", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("inf")}
    ]
}

ret = crowdtune.QuerySensitivityAnalysis(
    api_key = api_key,
    tuning_problem_name = "IJ",
    problem_space = problem_space,
    input_task = [100,100,100])
print ("Task: 100,100,100")
print (ret)

