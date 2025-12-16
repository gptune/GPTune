import os
import os.path as osp
import argparse
import pickle
import numpy as np
from operator import itemgetter
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='susy_10Kn')
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument("-bmin", type=int, default=1, help ='minimum value for bandit budget')
    parser.add_argument("-bmax", type=int, default=8, help ='maximum value for bandit budget')
    parser.add_argument("-eta", type=int, default=2, help ='base value for bandit structure')
    parser.add_argument("-Nloop", type=int, default=1, help ='number of bandit loops')
    parser.add_argument('-expid', type=str, default='0')
    return parser.parse_args()


def main(args):
    summary = []
    my_source = f'./KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.txt'
    save_path = f'./KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}.pkl'
    GPTuneBand_source =  f'./KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{args.expid}_GPTuneBand_parsed.pkl'
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if (info[0] == 'Tuner:' and info[1] == "GPTuneBand"):
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split() 
                    line = f.readline()
                    result = pickle.load(open(GPTuneBand_source, "rb"))
                    results.append(result)
                    if int(args.ntask) > 1:
                        line = f.readline().split()
                summary.append(results)
                line = f.readline()
            elif (info[0] == 'Tuner:' and info[1] == "hpbandster"):
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split()
                    task = line[0][7:]
                    line = f.readline().strip("    Os  ")
                    data = [[float(y) for y in x.split(", ")] for x in re.split('\[\[|\]\]|\), \(|\(|\)', line) if len(x) > 2]
                    data = [y for y in data if y[1] < float("Inf")]
                    x = []
                    y = []
                    pre_fix = 0
                    max_num = -999
                    for info in data:
                        if info[0] > max_num:
                            max_num = info[0]
                    for info in data:
                        pre_fix += info[0]/max_num
                        if np.isclose(info[0], max_num):
                            x.append(pre_fix)
                            y.append(info[1])
                    results.append([tid, task,  [x, y]]) 
                    if int(args.ntask) > 1:
                        line = f.readline().split()
                summary.append(results)
                line = f.readline()
            else: # GPTune OpenTuner and TPE
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split() 
                    task = line[0][7:]
                    line = f.readline().strip('    Os  [ ]\n') 
                    history = [float(x) for x in re.split('\], \[', line)]
                    x = list(np.arange(1,len(history)+1))
                    results.append([tid, task, [x,history]])
                    if int(args.ntask) > 1:
                        line = f.readline().split() 
                summary.append(results)
                line = f.readline()
               
    print(summary[0])
    print(summary[1])
    print(summary[2])
    print(summary[3])
    print(summary[4])
    print("Results saved to", save_path)
    pickle.dump(summary, open(save_path, "wb"))  

if __name__ == "__main__":
    main(parse_args())

