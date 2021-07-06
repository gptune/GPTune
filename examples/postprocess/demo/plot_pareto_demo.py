#! /usr/bin/env python


################################################################################
import sys
import os
import pickle

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    Or1 = pickle.load(open("pso_pareto.pkl", "rb"))
    Or2 = pickle.load(open("nsga2_pareto.pkl", "rb"))
    fontsize=30
    fig = plt.figure(figsize=[12.8, 9.6])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.scatter(Or1[:,0],Or1[:,1], s=30, c='k',label='pso on product of EIs',alpha=0.7)
    plt.scatter(Or2[:,0],Or2[:,1], s=30, c='r',label='nsga2 on multiple EIs',alpha=0.7)
    plt.legend(fontsize=fontsize)
    plt.xlabel('y1',fontsize=fontsize)
    plt.ylabel('y2',fontsize=fontsize)
    plt.show(block=False)
    plt.pause(0.5)  
    fig.savefig('pareto_demo_MO.pdf')                

