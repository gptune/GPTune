#! /usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../../../autotune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../../../scikit-optimize/"))

from gptune import * # import all

import numpy as np

def load_surrogate_model():

    model_configurations = GetSurrogateModelConfigurations()

    print ("MODEL CONFIGURATIONS")
    print (model_configurations)

    print ("LOAD FUNCTION 1")

    model_function = LoadSurrogateModelFunction(tuning_configuration=model_configurations[0])

    giventask = [[10000, 10000]]
    " A quick validation"
    ret = model_function({
        "m": giventask[0][0],
        "n": giventask[0][1],
        "mb": 16,
        "nb": 16,
        "npernode": 5,
        "p": 27})
    print ("func return: ", ret)

    ret = model_function({
        "m": giventask[0][0],
        "n": giventask[0][1],
        "mb": 15,
        "nb": 16,
        "npernode": 6,
        "p": 21})
    print ("func return: ", ret)

    print ("LOAD FUNCTION 2")

    model_function = LoadSurrogateModelFunction(tuning_configuration=model_configurations[1])

    giventask = [[10000, 10000]]
    " A quick validation"
    ret = model_function({
        "m": giventask[0][0],
        "n": giventask[0][1],
        "mb": 16,
        "nb": 16,
        "npernode": 5,
        "p": 13})
    print ("func return: ", ret)

    ret = model_function({
        "m": giventask[0][0],
        "n": giventask[0][1],
        "mb": 1,
        "nb": 1,
        "npernode": 1,
        "p": 1})
    print ("func return: ", ret)

    ret = model_function({
        "m": giventask[0][0],
        "n": giventask[0][1],
        "mb": 15,
        "nb": 14,
        "npernode": 5,
        "p": 257})
    print ("func return: ", ret)

def sensitivity_analysis():

    model_data = {
      "hyperparameters": [
        0.03557837804052048,
        1383.6391344152842,
        23357.489156222015,
        0.03881096498041434,
        1.0,
        6.491900279806671,
        1e-05,
        4.402125594862267
      ],
      "model_stats": {
        "log_likelihood": -61.55737787701022,
        "neg_log_likelihood": 61.55737787701022,
        "gradients": [
          -7.457458444438187e-06,
          4.271681386698028e-09,
          1.9359733671412996e-11,
          -1.3786743483838748e-06,
          0.0,
          -1.2622095368897135e-08,
          0.00015362341574541808,
          -7.535544455095078e-08
        ],
        "iterations": 128
      },
      "task_parameters": [
        [
          10000,
          10000
        ]
      ],
      "function_evaluations": [
        "7efe85d2-b4ec-11eb-a944-a7c92100e8b1",
        "7efe9540-b4ec-11eb-a944-a7c92100e8b1",
        "7efe9fe0-b4ec-11eb-a944-a7c92100e8b1",
        "7efeaa26-b4ec-11eb-a944-a7c92100e8b1",
        "7efeb570-b4ec-11eb-a944-a7c92100e8b1",
        "7efec038-b4ec-11eb-a944-a7c92100e8b1",
        "7efeca2e-b4ec-11eb-a944-a7c92100e8b1",
        "7efed44c-b4ec-11eb-a944-a7c92100e8b1",
        "7efede60-b4ec-11eb-a944-a7c92100e8b1",
        "7efee874-b4ec-11eb-a944-a7c92100e8b1",
        "7efef27e-b4ec-11eb-a944-a7c92100e8b1",
        "7efefcec-b4ec-11eb-a944-a7c92100e8b1",
        "7eff06ec-b4ec-11eb-a944-a7c92100e8b1",
        "7eff1128-b4ec-11eb-a944-a7c92100e8b1",
        "7eff1c04-b4ec-11eb-a944-a7c92100e8b1",
        "7eff2848-b4ec-11eb-a944-a7c92100e8b1",
        "7eff32ac-b4ec-11eb-a944-a7c92100e8b1",
        "7eff3cac-b4ec-11eb-a944-a7c92100e8b1",
        "7eff4698-b4ec-11eb-a944-a7c92100e8b1",
        "7eff50d4-b4ec-11eb-a944-a7c92100e8b1",
        "7eff5af2-b4ec-11eb-a944-a7c92100e8b1",
        "7eff6556-b4ec-11eb-a944-a7c92100e8b1",
        "7eff6f4c-b4ec-11eb-a944-a7c92100e8b1",
        "7eff794c-b4ec-11eb-a944-a7c92100e8b1",
        "7eff8374-b4ec-11eb-a944-a7c92100e8b1"
      ],
      "input_space": [
        {
          "name": "m",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 128,
          "upper_bound": 10000
        },
        {
          "name": "n",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 128,
          "upper_bound": 10000
        }
      ],
      "parameter_space": [
        {
          "name": "mb",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 1,
          "upper_bound": 16
        },
        {
          "name": "nb",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 1,
          "upper_bound": 16
        },
        {
          "name": "npernode",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 0,
          "upper_bound": 5
        },
        {
          "name": "p",
          "transformer": "normalize",
          "type": "int",
          "lower_bound": 1,
          "upper_bound": 32
        }
      ],
      "output_space": [
        {
          "name": "r",
          "transformer": "identity",
          "type": "real",
          "lower_bound": float("-Inf"),
          "upper_bound": float("Inf")
        }
      ],
      "modeler": "Model_LCM",
      "objective_id": 0,
      "time": {
        "tm_year": 2021,
        "tm_mon": 5,
        "tm_mday": 14,
        "tm_hour": 12,
        "tm_min": 42,
        "tm_sec": 35,
        "tm_wday": 4,
        "tm_yday": 134,
        "tm_isdst": 1
      },
      "uid": "88695f66-b4ec-11eb-a944-a7c92100e8b1"
    }

    SensitivityAnalysis(model_data=model_data, task_parameters=[10000,10000], num_samples=100)

    return

def sensitivity_analysis2():

    meta_dict = {
        "tuning_problem_name":"PDGEQRF",
        "task_parameters":[[10000,10000]],
        "input_space": [
          {
            "name": "m",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 128,
            "upper_bound": 10000
          },
          {
            "name": "n",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 128,
            "upper_bound": 10000
          }
        ],
        "parameter_space": [
          {
            "name": "mb",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 1,
            "upper_bound": 16
          },
          {
            "name": "nb",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 1,
            "upper_bound": 16
          },
          {
            "name": "npernode",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 0,
            "upper_bound": 5
          },
          {
            "name": "p",
            "type": "int",
            "transformer": "normalize",
            "lower_bound": 1,
            "upper_bound": 32
          }
        ],
        "output_space": [
          {
            "name": "r",
            "type": "real",
            "transformer": "identity"
          }
        ],
        "loadable_machine_configurations": {
          "Cori" : {
            "haswell": {
              "nodes":[i for i in range(1,65,1)],
              "cores":32
            },
            "knl": {
              "nodes":[i for i in range(1,65,1)],
              "cores":68
            }
          }
        },
        "loadable_software_configurations": {
          "openmpi": {
            "version_from":[4,0,1],
            "version_to":[5,0,0]
          },
          "scalapack":{
            "version_split":[2,1,0]
          },
          "gcc": {
            "version_split": [8,3,0]
          }
        }
    }

    model_data = LoadSurrogateModelData(meta_path=None, meta_dict=meta_dict)
    print (model_data)
    SensitivityAnalysis(model_data=model_data, task_parameters=[10000,10000], num_samples=100)

    return

if __name__ == "__main__":
    #load_surrogate_model()
    #sensitivity_analysis()
    sensitivity_analysis2()
