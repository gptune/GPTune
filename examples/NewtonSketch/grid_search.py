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

import sys
import os
import logging
import argparse
import numpy as np
import time
import json

sys.path.insert(0, os.path.abspath(__file__ + "/newtonsketch/"))

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores
from sklearn.kernel_approximation import RBFSampler
from solvers_lr import LogisticRegression
import generate_dataset

def objectives(point):
    dataset = point['dataset']
    sketch = point['sketch']
    n = point["n"]
    d = point["d"]
    m = int(d*point['sketch_size'])
    nnz = point['sparsity_parameter']*point["d"]/point["n"]
    lambd = point['lambd']
    error_threshold = point['error_threshold']
    niter = point['niter']

    print ("Dataset: ", dataset, "n: ", n, "d: ", d, "sketch: ", sketch, "lambda: ", lambd, "m: ", m, "nnz: ", nnz, "error_threshold: ", error_threshold, "niter: ", niter)

    times_spent = []
    for i in range(niter):
        _, losses_, times_ = lreg.ihs_tuning(sketch_size=m, sketch=sketch, nnz=nnz, error_threshold=error_threshold)

        print (losses_)
        print (times_)

        time_spent = times_[-1]
        times_spent.append(time_spent)
        loss_final = losses_[-1]

    return [times_spent]

#def cst1(sketch_size, n, d):
#    num_sketch_rows = int(d*sketch_size)
#    return num_sketch_rows >= 1 and num_sketch_rows <= n
#
#def cst2(sparsity_parameter, n, d):
#    nnzs_per_row = int(sparsity_parameter*d/n*n)
#    return nnzs_per_row >= 1 and nnzs_per_row <= n

def main():

    global nodes
    global cores

    dataset = "epsilon_normalized_20Kn_spread"
    nrun = 100

    global A, b, lreg

    A, b = generate_dataset.load_data('epsilon_normalized_20Kn', option="spread")
    lambd = 1e-4
    error_threshold = 1e-6

    n, d = A.shape
    niter = 1

    lreg = LogisticRegression(A, b, lambd)
    x, losses = lreg.solve_exactly(n_iter=20, eps=1e-15)

    for sketch_size_idx in range(10):
        for sparsity_parameter_idx in range(10):
            json_data_arr = []
            with open("grid_search.json", "r") as f_in:
                json_data_arr = json.load(f_in)

            skip_eval = False
            for json_data in json_data_arr:
                if sketch_size_idx == json_data["sketch_size_idx"] and sparsity_parameter_idx == json_data["sparsity_parameter_idx"]:
                    skip_eval = True
            if skip_eval == True:
                continue

            sketch_size = (1./d) + (2.0-1./d)/11.0*float(sketch_size_idx+1)
            sparsity_parameter = (1./d) + (n/d-1./d)/11.0*float(sparsity_parameter_idx+1)

            point = {
                "dataset": "epsilon_normalized_20Kn_spread",
                "sketch": "less_sparse",
                "n": n,
                "d": d,
                "sketch_size":sketch_size,
                "sparsity_parameter":sparsity_parameter,
                "lambd":lambd,
                "error_threshold":error_threshold,
                "niter":1,
            }
            outputs = objectives(point)

            output_dict = {
                "sketch_size_idx": sketch_size_idx,
                "sparsity_parameter_idx": sparsity_parameter_idx,
                "sketch_size": sketch_size,
                "sparsity_parameter": sparsity_parameter,
                "wall_clock_time": np.average(outputs[0])
            }

            json_data_arr.append(output_dict)
            with open("grid_search.json", "w") as f_out:
                json.dump(json_data_arr, f_out, indent=2)

if __name__ == "__main__":
    main()
