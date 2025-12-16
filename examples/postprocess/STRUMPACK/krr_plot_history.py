import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from operator import itemgetter
from argparse import Namespace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='susy_10Kn')
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument("-bmin", type=int, default=1, help ='minimum value for bandit budget')
    parser.add_argument("-bmax", type=int, default=8, help ='maximum value for bandit budget')
    parser.add_argument("-eta", type=int, default=2, help ='base value for bandit structure')
    parser.add_argument("-Nloop", type=int, default=1, help ='number of bandit loops')
    parser.add_argument('-baseline', type=float, default=None)
    parser.add_argument('-explist', nargs='+', help='a list of repeated experiment for error bar plots', required=True)
    parser.add_argument('-deleted_tuners', nargs='+', help='a list of tuners not plotting', default=None)
    return parser.parse_args()

def gen_source(args, expid=None):
    my_source = f'./KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{expid}.pkl'
    return my_source


def data_process_single_exp(args, expid=None):
    my_source = gen_source(args, expid=expid)
    results_summary = pickle.load(open(my_source, "rb"))
    GPTuneBand_results = results_summary[0]
    GPTune_results = results_summary[1]
    HpBandster_results = results_summary[2]
    TPE_results = results_summary[3]
    OpenTuner_results = results_summary[4]
    # assert GPTuneBand_results[0] == 'GPTuneBand'
    # assert GPTune_results[0] == 'GPTune'
    # assert HpBandster_results[0] == 'hpbandster'
    # assert TPE_results[0] == 'TPE'
    # assert OpenTuner_results[0] == 'opentuner'
    
    colors = ['#C33734', '#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4']
    linestyles = ['solid', 'dashed', 'dashdot', 'dashdot', 'dotted']
    data_summary = []
    tuners = []
    for item in results_summary:
        tuners.append(item[0])
        data_summary.append(item[1][2])
        
    data_summary_new = []
    for i,data in enumerate(data_summary):
        data_summary_new.append([data[0], historical_best(data[1])])
        
    return data_summary_new, tuners, colors, linestyles

def data_process(args):
    if len(args.explist) == 1:
        return data_process_single_exp(args, expid=args.explist[0])
    else:
        data_summary = []
        for expid in args.explist:
            data_summary_cur, tuners, colors, linestyles = data_process_single_exp(args, expid=expid)
            data_summary.append(data_summary_cur)
        return data_summary, tuners, colors, linestyles
    

def historical_best(data):
    for i in range(len(data)-1):
        data[i+1] = data[i] if data[i] < data[i+1] else data[i+1]
    return data

# i: sorted i-th task
def plot_single(args, data_summary, tuners, colors, linestyles, expid=None):
    my_source = gen_source(args, expid=expid)
    filename = os.path.splitext(os.path.basename(my_source))[0]

    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if args.baseline != None:
        ax.plot(data_summary[0], 
                args.fmin*np.ones(len(data_summary[0])), 
                'k--', label="True")
    for i,data in enumerate(data_summary):
        ax.plot(data[0], data[1], color=colors[i], 
                label=tuners[i], linestyle=linestyles[i])

    ax.legend(fontsize=8)
    savepath = os.path.join("./", f"Tuning_history_{filename}.pdf")    
    plt.savefig(savepath)
    print("Figure saved: ", savepath)

def plot_history(args,  data_summary, tuners, colors, 
                 linestyles, baseline=None,
                 deleted_tuners=None):
    if len(args.explist) == 1:
        plot_single(args, data_summary, tuners, colors, linestyles, 
                    expid=args.explist[0])
    else:
        N_tuners = len(tuners)
        N_exps = len(args.explist)
        bugets_set = []
        mean_results_set = []
        std_results_set = []
        for i in range(N_tuners):
            bugets_set.append(data_summary[0][i][0])
            results_set=[]
            # print("Tuner:", tuners[i])
            for j in range(N_exps):
                results_set.append(data_summary[j][i][1])
            # print("results set:")
            # print(results_set)
            mean_results_set.append(np.mean(results_set, axis=0))
            std_results_set.append(np.std(results_set, ddof=0,axis=0))
            # print("Mean and std")
            # print(bugets_set[i])
            # print(mean_results_set[i])
            # print(std_results_set[i])
            # print()
        expname = '-'.join(args.explist)
        my_source = gen_source(args, expid=expname)
        filename = os.path.splitext(os.path.basename(my_source))[0]
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if baseline != None:
            ax.plot(data_summary[0][0][0], 
                    args.fmin*np.ones(len(data_summary[0][0][0])), 
                    'k--', label="True")

        for i in range(N_tuners):
            if args.deleted_tuners == None or tuners[i] not in args.deleted_tuners:
                ax.plot(bugets_set[i], 
                                mean_results_set[i],
                                color=colors[i], 
                                label=tuners[i], 
                                linestyle=linestyles[i])
                ax.fill_between(bugets_set[i], 
                                mean_results_set[i]+std_results_set[i], 
                                mean_results_set[i]-std_results_set[i],
                                color=colors[i], 
                                alpha=0.5)
        ax.legend(fontsize=8)
        savepath = os.path.join("./", f"Tuning_history_{filename}.pdf")    
        plt.savefig(savepath)
        print("Figure saved: ", savepath)

def main(args):
    print()
    print("plotting args:", args)
    data_summary, tuners, colors, linestyles = data_process(args)
    plot_history(args, data_summary, tuners, colors, linestyles)

    
if __name__ == "__main__":
    main(parse_args())