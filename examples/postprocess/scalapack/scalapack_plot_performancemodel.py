import os
import os.path as osp
import argparse
import pickle 
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
    parser.add_argument('--mmax', type=int, default=100, help='maximum row dimension')
    parser.add_argument('--nmax', type=int, default=100, help='minimum column dimension')
    parser.add_argument('--nodes', type=int, default=100, help='number of compute nodes')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument('--nrun1', type=int, default=10, help='number of runs')
    parser.add_argument('--nrun2', type=int, default=10, help='number of runs')
    parser.add_argument('--nrun3', type=int, default=10, help='number of runs')
    parser.add_argument('--exp', type=str,default='tuners',help='the experiment tags (tuners, models)')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./{args.exp}_scalapack_mmax{args.mmax}_nmax{args.nmax}_nodes{args.nodes}_ntask{args.ntask}.pkl"
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    ntask = len(results_summary)
    assert ntask == args.ntask
    PerModelvsNoModel1 = []
    PerModelvsNoModel2 = []
    PerModelvsNoModel3 = []
    size_set = []
    for i in range(ntask):
        task_current = results_summary[i]
        results_PerModel = task_current[3][0]
        results_NoModel = task_current[3][1]
        assert results_PerModel[0] == "PerModel"
        assert results_NoModel[0] == "NoModel"
        idx_nrun = task_current[2].index(args.nrun1)
        PerModelvsNoModel1.append(results_NoModel[1][idx_nrun]/results_PerModel[1][idx_nrun])
        idx_nrun = task_current[2].index(args.nrun2)
        PerModelvsNoModel2.append(results_NoModel[1][idx_nrun]/results_PerModel[1][idx_nrun])
        idx_nrun = task_current[2].index(args.nrun3)
        PerModelvsNoModel3.append(results_NoModel[1][idx_nrun]/results_PerModel[1][idx_nrun])        
    return PerModelvsNoModel1, PerModelvsNoModel2, PerModelvsNoModel3

def plot(data1, data2, data3, args):
    my_source = gen_source(args)
    ntask = len(data1)
    
    font_mag=matplotlib.font_manager.FontProperties(family=font['family'], weight=font['weight'], size=font['size'])


    p1 = len([x for x in data1 if x >= 1])
    p2 =  len([x for x in data2 if x >= 1])
    p3 =  len([x for x in data3 if x >= 1])
    
    # plot    
    # plt.clf()
    x = np.arange(1, ntask+1)
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    # ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune, {p1}(>=1), {p3}(<0.5)')
    # ax.bar(x + width/2, data2, width, label=f'HpBandster/GPTune, {p2}(>=1), {p4}(<0.5)')
    ax.bar(x - width, data1, width, color='tab:blue', label=f'NS={args.nrun1}, {p1}(>=1)')
    ax.bar(x , data2, width, color='tab:orange', label=f'NS={args.nrun2}, {p2}(>=1)')
    ax.bar(x + width, data3, width, color='tab:brown', label=f'NS={args.nrun3}, {p3}(>=1)')
    ax.plot([1- width*3/2,ntask+width*3/2], [1, 1], c='black', linestyle=':')
    # ax.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    ax.set_ylabel('Ratio of objective minimum',fontdict=font)
    ax.set_xlabel('Task ID',fontdict=font)
    # ax.set_title(f'[m, n] in [{args.mmax}, {args.nmax}]',fontdict=font)   
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(prop=font_mag,loc='lower left',ncol=1)
    fig.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    fig.savefig(os.path.join("./plots", f"{filename}.pdf"))

# def plot_histogram(data1, args):
#     my_source = gen_source(args)
#     nrun = args.nrun
#     ntask = len(data1)
    
#     plt.clf()
#     fig, ax1 = plt.subplots()
#     ax1.hist(data1, bins=np.arange(0, int(np.ceil(max(data1)))+1, 0.5), weights=np.ones(ntask) / ntask, 
#              label=f'NoModel/PerModel, range: [{min(data1):.2f}, {max(data1):.2f}]', color='#1f77b4')
#     ax1.plot([1, 1], [0, 1], c='black', linestyle=':')
#     ax1.legend(fontsize=8)
#     ax1.set_title(f'[m, n] in [{args.mmax}, {args.nmax}], nrun = {nrun}')
#     ax1.set_ylabel('Fraction')
#     ax1.set_xlabel('Ratio of best performance')
#     fig.tight_layout()
#     filename = os.path.splitext(os.path.basename(my_source))[0]
#     plt.ion()
#     plt.show()
#     plt.pause(0.001)
#     # input("Press [enter] to continue.")    
#     fig.savefig(os.path.join("./plots", f"hist_{filename}_nrun{nrun}.pdf"))
    
    
    
def main(args):
    PerModelvsNoModel1, PerModelvsNoModel2, PerModelvsNoModel3 = data_process(args)
    print(PerModelvsNoModel1)
    print(PerModelvsNoModel2)
    print(PerModelvsNoModel3)
    plot(PerModelvsNoModel1,PerModelvsNoModel2, PerModelvsNoModel3, args)
    # plot_histogram(OpenTunervsGPTune, args)
    
if __name__ == "__main__":
    main(parse_args())