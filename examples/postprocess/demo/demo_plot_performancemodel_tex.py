import os
import os.path as osp
import argparse
import pickle 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


font = {'family': 'serif',
    'weight': 'normal',
    'size': 16,
    }
matplotlib.rc('font', **font)   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100, help='number of compute nodes')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument('--nrun1', type=int, default=10, help='number of runs')
    parser.add_argument('--nrun2', type=int, default=10, help='number of runs')
    parser.add_argument('--nrun3', type=int, default=10, help='number of runs')
    parser.add_argument('--exp', type=str,default='tuners',help='the experiment tags (tuners, models)')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./{args.exp}_demo_nodes{args.nodes}_ntask{args.ntask}.pkl"
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    ntask = len(results_summary)
    assert ntask == args.ntask
    PerModelvsNoModel1 = []
    PerModelvsNoModel2 = []
    PerModelvsNoModel3 = []
    PerModelvsTrue1 = []
    PerModelvsTrue2 = []
    PerModelvsTrue3 = []    
    size_set = []
    for i in range(ntask):
        task_current = results_summary[i]
        results_PerModel = task_current[3][0]
        results_NoModel = task_current[3][1]
        results_True = task_current[3][2]
        assert results_PerModel[0] == "PerModel"
        assert results_NoModel[0] == "NoModel"
        assert results_True[0] == "True "
        idx_nrun = task_current[2].index(args.nrun1)
        PerModelvsNoModel1.append((1+results_NoModel[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))
        PerModelvsTrue1.append((1+results_True[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))        
        idx_nrun = task_current[2].index(args.nrun2)
        PerModelvsNoModel2.append((1+results_NoModel[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))
        PerModelvsTrue2.append((1+results_True[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))         
        idx_nrun = task_current[2].index(args.nrun3)
        PerModelvsNoModel3.append((1+results_NoModel[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))  
        PerModelvsTrue3.append((1+results_True[1][idx_nrun])/(1+results_PerModel[1][idx_nrun]))    
    return PerModelvsNoModel1, PerModelvsNoModel2, PerModelvsNoModel3, PerModelvsTrue1, PerModelvsTrue2, PerModelvsTrue3

def plot(data1, data2, data3, data10,data20,data30, args):
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
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.005)
    # ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune, {p1}(>=1), {p3}(<0.5)')
    # ax.bar(x + width/2, data2, width, label=f'HpBandster/GPTune, {p2}(>=1), {p4}(<0.5)')
    ax1.bar(x - width, data1, width, color='tab:blue', label=r'$\epsilon_{tot}$'+f'={args.nrun1}, {p1}($\ge$1)')
    ax1.bar(x , data2, width, color='tab:orange', label=r'$\epsilon_{tot}$'+f'={args.nrun2}, {p2}($\ge$1)')
    ax1.bar(x + width, data3, width, color='tab:brown', label=r'$\epsilon_{tot}$'+f'={args.nrun3}, {p3}($\ge$1)')
    ax1.plot([1- width*3/2,ntask+width*3/2], [1, 1], c='black', linestyle=':')
    # ax.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    ax1.set_ylabel('Ratio vs no model',fontdict=font)
    # ax1.set_xlabel('Task ID',fontdict=font)
    # ax.set_title(f'[m, n] in [{args.mmax}, {args.nmax}]',fontdict=font)   
    ax1.set_xticks(x)
    ax1.set_xticklabels(x)
    ax1.legend(prop=font_mag,loc='lower right',ncol=1)


    ax2.bar(x - width, data10, width, color='tab:blue', label=r'$\epsilon_{tot}$'+f'={args.nrun1}')
    ax2.bar(x , data20, width, color='tab:orange', label=r'$\epsilon_{tot}$'+f'={args.nrun2}')
    ax2.bar(x + width, data30, width, color='tab:brown', label=r'$\epsilon_{tot}$'+f'={args.nrun3}')
    ax2.plot([1- width*3/2,ntask+width*3/2], [1, 1], c='black', linestyle=':')
    # ax.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    ax2.set_ylabel('Ratio vs true',fontdict=font)
    ax2.set_xlabel('Task ID',fontdict=font)
    # ax.set_title(f'[m, n] in [{args.mmax}, {args.nmax}]',fontdict=font)   
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.legend(prop=font_mag,loc='lower right',ncol=1)


    fig.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    fig.savefig(os.path.join("./plots", f"{filename}.pdf"))

    
def main(args):
    PerModelvsNoModel1, PerModelvsNoModel2, PerModelvsNoModel3,PerModelvsTrue1,PerModelvsTrue2,PerModelvsTrue3 = data_process(args)
    print(PerModelvsNoModel1)
    print(PerModelvsNoModel2)
    print(PerModelvsNoModel3)
    print(PerModelvsTrue1)
    print(PerModelvsTrue2)
    print(PerModelvsTrue3)
    plot(PerModelvsNoModel1,PerModelvsNoModel2, PerModelvsNoModel3,PerModelvsTrue1,PerModelvsTrue2,PerModelvsTrue3, args)
    # plot_histogram(OpenTunervsGPTune, args)
    
if __name__ == "__main__":
    main(parse_args())