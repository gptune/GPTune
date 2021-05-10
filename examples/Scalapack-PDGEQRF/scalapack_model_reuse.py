#! /usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all

import numpy as np

def main():

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

if __name__ == "__main__":
    main()
