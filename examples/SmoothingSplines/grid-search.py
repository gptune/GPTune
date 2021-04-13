#! /usr/bin/env python

import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import time
import os
import math

def compute_rsquared(\
    baseline,\
    prediction,\
    total_points):

    mean_y = 0.0
    for i in range(total_points):
        mean_y += baseline[i]
    mean_y /= float(total_points)

    SS_res = 0.0
    for i in range(total_points):
        SS_res += math.pow((baseline[i] - prediction[i]), 2)

    SS_tot = 0.0
    for i in range(total_points):
        SS_tot += math.pow((baseline[i] - mean_y), 2)

    Rsquared = 1.0 - SS_res/SS_tot

    return Rsquared

def compute_arsquared(\
    baseline,\
    prediction,\
    total_points):

    mean_y = 0.0
    for i in range(total_points):
        mean_y += baseline[i]
    mean_y /= float(total_points)

    SS_res = 0.0
    for i in range(total_points):
        SS_res += math.pow((baseline[i] - prediction[i]), 2)

    SS_tot = 0.0
    for i in range(total_points):
        SS_tot += math.pow((baseline[i] - mean_y), 2)

    Rsquared = 1.0 - SS_res/SS_tot

    n = total_points
    p = 1
    Adjusted = 1.0 - (1.0-Rsquared)*float(n-1)/float(n-p-1)

    return Adjusted

def run_grid_search(dataset_name, X_train, Y_train, X_test, Y_test):

    print ("dataset_name")
    print (dataset_name)

    stats = {
        "regression_time":[],
        "in_test_time":[],
        "out_test_time":[],
        "in_mse":[],
        "in_r2":[],
        "in_ar2":[],
        "out_mse":[],
        "out_r2":[],
        "out_ar2":[]
    }

    os.system("mkdir -p grid-search.db")
    with open("grid-search.db/"+dataset_name+".log", "a") as f_out:
        f_out.write("NKnots,Lambda,RegressionTime,InTestTime,InMSE,InR2,InAR2,OutTestTime,OutMSE,OutR2,OutAR2\n")
        for k in [4,5,6,7,8,9,10,11]:
            for l in range(21):
                lambda1 = float(l)/20.0

                print ("Grid-search Dataset: "+str(dataset_name)+" K: "+str(k)+" L:"+str(lambda1))
                f_out.write(str(k)+","+str(lambda1)+",")

                r_y = robjects.FloatVector(Y_train)
                r_x = robjects.FloatVector(X_train)

                try:
                    t1 = time.time_ns()
                    r_smooth_spline = robjects.r['smooth.spline']
                    spline1 = r_smooth_spline(x=r_x, y=r_y, spar=lambda1, nknots=k)
                    t2 = time.time_ns()

                    r_time = (t2-t1)/1e9
                    stats["regression_time"].append(r_time)
                    f_out.write(str(r_time)+",")
                except:
                    stats["regression_time"].append("-")
                    f_out.write("\n")
                    continue

                t1 = time.time_ns()
                Y_train_spline = np.array(robjects.r['predict'](spline1,robjects.FloatVector(X_train)).rx2('y'))
                in_mse = np.mean((np.array(Y_train)-np.array(Y_train_spline))**2)
                in_R2 = compute_rsquared(Y_train, Y_train_spline, len(Y_train))
                in_AR2 = compute_arsquared(Y_train, Y_train_spline, len(Y_train))
                t2 = time.time_ns()

                in_test_time = (t2-t1)/1e9
                stats["in_test_time"].append(in_test_time)
                stats["in_mse"].append(in_mse)
                stats["in_r2"].append(in_R2)
                stats["in_ar2"].append(in_AR2)

                t1 = time.time_ns()
                Y_test_spline = np.array(robjects.r['predict'](spline1,robjects.FloatVector(X_test)).rx2('y'))
                out_mse = np.mean((np.array(Y_test)-np.array(Y_test_spline))**2)
                out_R2 = compute_rsquared(Y_test, Y_test_spline, len(Y_test))
                out_AR2 = compute_arsquared(Y_test, Y_test_spline, len(Y_test))
                t2 = time.time_ns()

                out_test_time = (t2-t1)/1e9
                stats["out_test_time"].append(out_test_time)
                stats["out_mse"].append(out_mse)
                stats["out_r2"].append(out_R2)
                stats["out_ar2"].append(out_AR2)

                f_out.write(str(in_test_time)+","+str(in_mse)+","+str(in_R2)+","+str(in_AR2)+","+str(out_test_time)+","+str(out_mse)+","+str(out_R2)+","+str(out_AR2)+"\n")

    print (stats)

    return

def main():
    #for task in [1.0, 2.0, 3.0, 4.0, 5.0]:
    for task in [1.0]:
        num_samples = 10000000
        dataset = "gptune-demo-"+str(task)+"-"+str(num_samples)
        #dataset = sys.argv[1]

        X_train = []
        Y_train = []

        trainset = dataset+"-train"
        with open("datagen/"+trainset, "r") as f_in:
            for dataline in f_in.readlines():
                data = dataline.split(",")
                X_train.append(float(data[0]))
                Y_train.append(float(data[1]))

        X_test = []
        Y_test = []

        testset = dataset+"-test"
        with open("datagen/"+testset, "r") as f_in:
            for dataline in f_in.readlines():
                data = dataline.split(",")
                X_test.append(float(data[0]))
                Y_test.append(float(data[1]))

        run_grid_search(dataset, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()

