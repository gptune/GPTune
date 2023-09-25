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

def run_grid_search(dataset_name, size, X_train, Y_train, X_test, Y_test):

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

        k_unit = int(size/50)
        for k in range(k_unit, size, k_unit):
            for lambda1 in [0,0.1,0.01,10**-3,10**-4,10**-5,10**-6,10**-7,10**-8,10**-9,10**-10,1]:

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

def run_synthetic_test():

    for t in [0.5,1.0,2.0,3.0,4.0,5.0]:
        for size in [10000,100000]:
            for v in [0.01, 0.05, 0.1]:
                dataset = "gptune-demo-"+str(t)+"-"+str(size)+"-"+str(v)
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

                run_grid_search(dataset, size, X_train, Y_train, X_test, Y_test)

    return

def run_grid_search_winequality(dataset_name, size, X_train, Y_train, X_test, Y_test):

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

        #k_unit = int(size/50)
        #for k in range(k_unit, size, k_unit):
        for k in [4,5,6,7,8,9,10,11,12,13]:
            for lambda1 in [0,0.1,0.01,10**-3,10**-4,10**-5,10**-6,10**-7,10**-8,10**-9,10**-10,1]:

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

def run_winequality_test():

    attributes = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]

    for attribute in attributes:
        dataset = "winequality-"+attribute

        idx = attributes.index(attribute)

        X_train = []
        Y_train = []

        with open("winequality/wine_train.txt", "r") as f_in:
            f_in.readline()
            for dataline in f_in.readlines():
                data = dataline.split(" ")
                X_train.append(float(data[idx]))

        with open("winequality/score_train.txt", "r") as f_in:
            f_in.readline()
            for dataline in f_in.readlines():
                data = dataline.split(" ")
                Y_train.append(float(data[0]))

        X_test = []
        Y_test = []

        with open("winequality/wine_test.txt", "r") as f_in:
            f_in.readline()
            for dataline in f_in.readlines():
                data = dataline.split(" ")
                X_test.append(float(data[idx]))

        with open("winequality/score_test.txt", "r") as f_in:
            f_in.readline()
            for dataline in f_in.readlines():
                data = dataline.split(" ")
                Y_test.append(float(data[0]))

        run_grid_search_winequality(dataset, len(X_test), X_train, Y_train, X_test, Y_test)

    return

def main():

    #run_synthetic_test()
    run_winequality_test()

    return

if __name__ == "__main__":
    main()

