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

import sys, os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

import math

mmin = 128
mmax = 1000
nmin = 128
nmax = 1000
nodes = 1
cores = 8
bunit = 8
nprocmin_pernode = 1

problem_space = {
    "input_space": [
        {"name":"m", "type":"integer", "transformer":"normalize", "lower_bound":mmin, "upper_bound": mmax},
        {"name":"n", "type":"integer", "transformer":"normalize", "lower_bound":nmin, "upper_bound": nmax}
    ],
    "constants": {
        "nodes": nodes,
        "cores": cores,
        "bunit": bunit
    },
    "parameter_space": [
        {"name":"mb", "type":"integer", "transformer": "normalize", "lower_bound":1, "upper_bound": 16},
        {"name":"nb", "type":"integer", "transformer": "normalize", "lower_bound":1, "upper_bound": 16},
        {"name":"lg2npernode", "type":"integer", "transformer": "normalize", "lower_bound":int(math.log2(nprocmin_pernode)), "upper_bound":int(math.log2(cores))},
        {"name":"p", "type":"integer", "transformer": "normalize", "lower_bound":1, "upper_bound": nodes*cores}
    ],
    "output_space": [
        {"name":"r", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("inf")}
    ]
}

print ("Run Sensitivity Analysis")

# Sensitivity analysis interface, reading DB from the given DB file path
import gptune
ret = gptune.SensitivityAnalysis(
    problem_space = problem_space,
    modeler="Model_GPy_LCM",
    method="Sobol",
    input_task = [1000,1000],
    historydb_path = "gptune.db/PDGEQRF.json",
    num_samples=512)
print (ret)

'''
# Sensitivity analysis interface, reading DB from the given tuning problem name
import gptune
ret = gptune.SensitivityAnalysis(
    problem_space = problem_space,
    modeler="Model_GPy_LCM",
    method="Sobol",
    input_task = [1000,1000],
    tuning_problem_name = "PDGEQRF", # read function evaluations from gptune.db/PDGEQRF.json
    num_samples=512)
print (ret)
'''

'''
# Sensitivity analysis interface, by providing a list of the function evaluations
with open("gptune.db/PDGEQRF.json", "r") as f_in:
    import json
    function_evaluations = json.load(f_in)["func_eval"]
import gptune
ret = gptune.SensitivityAnalysis(
    problem_space = problem_space,
    modeler="Model_GPy_LCM",
    method="Sobol",
    input_task = [1000,1000],
    function_evaluations = function_evaluations,
    num_samples=512)
print (ret)
'''
