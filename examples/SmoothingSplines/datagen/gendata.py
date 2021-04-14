#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def objectives(t, x):

    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """

    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1

    return f

def gen_data(dataset="gptune-demo", seed = 0, task = 1.0, num_samples = 100, var = 0.1):

    np.random.seed(seed)

    mu, sigma = 0, var # mean and standard deviation

    x_list = np.random.uniform(low=0.0, high=1.0, size=num_samples)
    x_list.sort()

    y_list = [objectives(task, x_list[i]) for i in range(num_samples)]
    err_list = np.random.normal(mu, sigma, num_samples)
    y_list_err = [y_list[i] + err_list[i] for i in range(num_samples)]

    with open(dataset, "w") as f_out:
        for i in range(num_samples):
            f_out.write(str(x_list[i])+","+str(y_list_err[i])+","+str(y_list[i])+"\n")

    f = plt.figure()
    plt.plot(x_list, y_list_err, "-")
    plt.plot(x_list, y_list, "-")
    #plt.show()
    f.savefig(dataset+".pdf")

    return

def main():

    for t in [0.5,1.0,2.0,3.0,4.0,5.0]:
        for size in [10000,100000]:
            for v in [0.01, 0.05, 0.1]:
                gen_data(dataset="gptune-demo-"+str(t)+"-"+str(size)+"-"+str(v)+"-train", seed=0, task=t, num_samples=size, var=v)
                gen_data(dataset="gptune-demo-"+str(t)+"-"+str(size)+"-"+str(v)+"-test", seed=1, task=t, num_samples=size, var=v)

    return

if __name__ == "__main__":
    main()

