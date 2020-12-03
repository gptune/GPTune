import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


font = {'family': 'serif',
    'weight': 'normal',
    'size': 18,
    }
matplotlib.rc('font', **font)   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100, help='number of compute nodes')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    parser.add_argument('--exp', type=str,default='tuners',help='the experiment tags (tuners, models)')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./{args.exp}_superlu_dist_nodes{args.nodes}_ntask{args.ntask}.pkl"
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    ntask = len(results_summary)
    assert ntask == args.ntask
    OpenTunervsGPTune = []
    HpBandstervsGPTune = []
    GPTune_time = []
    Opentuner_time = []
    HpBandster_time = []
    size_set = []
    for i in range(ntask):
        task_current = results_summary[i]
        print(task_current)
        idx_nrun = task_current[1].index(args.nrun)
        results_GPTune = task_current[2][0]
        results_OpenTuner = task_current[2][1]
        results_HpBandster = task_current[2][2]
        assert results_GPTune[0] == "GPTune"
        assert results_OpenTuner[0] == "OpenTuner"
        assert results_HpBandster[0] == "Hpbandster"
        OpenTunervsGPTune.append(results_OpenTuner[1][idx_nrun]/results_GPTune[1][idx_nrun])
        HpBandstervsGPTune.append(results_HpBandster[1][idx_nrun]/results_GPTune[1][idx_nrun])
        GPTune_time.append(results_GPTune[1][idx_nrun])
        Opentuner_time.append(results_OpenTuner[1][idx_nrun])
        HpBandster_time.append(results_HpBandster[1][idx_nrun])
    return OpenTunervsGPTune, HpBandstervsGPTune, GPTune_time, Opentuner_time, HpBandster_time

def plot(data1, data2, args):
    my_source = gen_source(args)
    assert len(data1) == len(data2) 
    nrun = args.nrun
    ntask = len(data1)
    font_mag=matplotlib.font_manager.FontProperties(family=font['family'], weight=font['weight'], size=font['size'])

    
    p1 = len([x for x in data1 if x >= 1])
    p2 = len([x for x in data2 if x >= 1])
    p3 =  len([x for x in data1 if x < 0.5])
    p4 =  len([x for x in data2 if x < 0.5])
    
    # plot    
    # plt.clf()
    x = np.arange(1, ntask+1)
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    # ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune, {p1}(>=1), {p3}(<0.5)')
    # ax.bar(x + width/2, data2, width, label=f'HpBandster/GPTune, {p2}(>=1), {p4}(<0.5)')
    ax.bar(x - width/2, data1, width, color='tab:blue', label=f'OpenTuner/GPTune, {p1}(>=1)')
    ax.bar(x + width/2, data2, width, color='tab:orange', label=f'HpBandster/GPTune, {p2}(>=1)')
    ax.plot([1- width*3/2,ntask+width*3/2], [1, 1], c='black', linestyle=':')
    # ax.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    ax.set_ylabel('Ratio of best runtime',fontdict=font)
    ax.set_xlabel('Task ID',fontdict=font)
    ax.set_title(f'PARSEC matrices, NS = {nrun}',fontdict=font)   
    ax.set_xticks(x)
    ax.set_xticklabels(x,fontdict=font)
    ax.legend(prop=font_mag)
    ax.set_ylim([0, 2])
    fig.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")    
    fig.savefig(os.path.join("./plots", f"{filename}_nrun{nrun}.pdf"))

def plot_histogram(data1, data2, args):
    
     
    font_mag=matplotlib.font_manager.FontProperties(family=font['family'], weight=font['weight'], size=font['size'])    
    
    my_source = gen_source(args)
    assert len(data1) == len(data2) 
    nrun = args.nrun
    ntask = len(data1)
    
    # plt.clf()
    bins1 = np.arange(0.75, int(np.ceil(max(data1))), 0.25)
    bins2 = np.arange(0.75, int(np.ceil(max(data2))), 0.25)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    n, b, patches=ax1.hist(data1, bins=bins1, weights=np.ones(ntask) / ntask, 
             label=f'OpenTuner/GPTune', color='#1f77b4')
    ax1.plot([1, 1], [0, n.max()*1.2], c='black', linestyle=':')
    n, b, patches=ax2.hist(data2, bins=bins2 , weights=np.ones(ntask) / ntask, 
             label=f'HpBandster/GPTune', color='#ff7f0e')    
    ax2.plot([1, 1], [0, n.max()*1.2], c='black', linestyle=':')
    ax1.legend(prop=font_mag)
    ax2.legend(prop=font_mag)
    ax1.set_title(f'NS = {nrun}',fontdict=font)
    ax1.set_ylabel('Fraction',fontdict=font)
    ax2.set_ylabel('Fraction',fontdict=font)
    ax2.set_xlabel('Ratio of objective minimum',fontdict=font)
    # ax1.set_xticks(bins1)
    # ax1.set_xticklabels(bins1)
    # ax2.set_xticks(bins2)
    # ax2.set_xticklabels(bins2)

    fig.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")       
    fig.savefig(os.path.join("./plots", f"hist_{filename}_nrun{nrun}.pdf"))
    
    
def main(args):
    OpenTunervsGPTune, HpBandstervsGPTune, GPTune_time, Opentuner_time, HpBandster_time = data_process(args)
    print(OpenTunervsGPTune)
    print(HpBandstervsGPTune)
    plot(OpenTunervsGPTune, HpBandstervsGPTune, args)
    plot_histogram(OpenTunervsGPTune, HpBandstervsGPTune, args)
    
if __name__ == "__main__":
    main(parse_args())