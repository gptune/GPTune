#! /usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all

import numpy as np

def main():

    model_function = LoadSurrogateModelFunction()

    giventask = [[4000, 4000]]
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

if __name__ == "__main__":
    main()
