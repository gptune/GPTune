#! /usr/bin/env python

import sys
import json

if __name__ == "__main__":

    merged_json_data = {}

    merged_json_data["tuning_problem_name"] = "GPTune-Demo"
    merged_json_data["surrogate_model"] = []
    merged_json_data["func_eval"] = []

    nargs = len(sys.argv)

    print ("nargs: ", nargs)

    for i in range(1, nargs, 1):

        with open(sys.argv[i], "r") as f_in:
            json_data = json.load(f_in)

            for surrogate_model in json_data["surrogate_model"]:
                merged_json_data["surrogate_model"].append(surrogate_model)
            for func_eval in json_data["func_eval"]:
                merged_json_data["func_eval"].append(func_eval)

    with open("db.out", "w") as f_out:

        json.dump(merged_json_data, f_out, indent=2)

