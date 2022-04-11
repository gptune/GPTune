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

# Can get an access token at https://gptune.lbl.gov/account/access-tokens/
# Keep your access tokens credential

import os
api_key = os.getenv("CROWDTUNE_API_KEY")

import crowdtune

problem_space = {
    "input_space": [
        {"name":"mx", "type":"integer", "transformer":"normalize", "lower_bound":5, "upper_bound":6},
        {"name":"my", "type":"integer", "transformer":"normalize", "lower_bound":7, "upper_bound":8},
        {"name":"lphi", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":3}
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

# Run a sensitivity analysis (SA)
ret = crowdtune.QuerySensitivityAnalysis(
    api_key = api_key,
    tuning_problem_name = "NIMROD_slu3d",
    problem_space = problem_space,
    configuration_space = configuration_space,
    modeler = "Model_GPy_LCM",
    method = "Sobol",
    input_task = [5,7,1])
print ("Sobol analysis result for task {mx:5,my:7,lphi:1}")
print (ret) # print SA result

problem_space = {
    "input_space": [
        {"name":"mx", "type":"integer", "transformer":"normalize", "lower_bound":6, "upper_bound":7},
        {"name":"my", "type":"integer", "transformer":"normalize", "lower_bound":8, "upper_bound":9},
        {"name":"lphi", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":3}
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
        {"name":"time", "type":"real", "transformer":"identity", "lower_bound":0, "upper_bound":999.9}
    ]
}
configuration_space = {}

# Run a sensitivity analysis (SA)
ret = crowdtune.QuerySensitivityAnalysis(
    api_key = api_key,
    tuning_problem_name = "NIMROD_slu3d",
    problem_space = problem_space,
    configuration_space = configuration_space,
    modeler = "Model_GPy_LCM",
    method = "Sobol",
    input_task = [6,8,1])
print ("Sobol analysis result for task {mx:6,my:8,lphi:1}")
print (ret) # print SA result

