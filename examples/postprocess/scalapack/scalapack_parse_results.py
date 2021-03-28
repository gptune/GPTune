import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmax', type=int, default=100, help='maximum row dimension')
    parser.add_argument('--nmax', type=int, default=100, help='minimum column dimension')
    parser.add_argument('--nodes', type=int, default=100, help='number of compute nodes')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument('--exp', type=str,default='tuners',help='the experiment tags (tuners, models)')
    return parser.parse_args()

def get_results_from_line(line):
    info = line.split("\t")
    tuner = info[0]
    results = list(map(lambda x: float(x.split()[0]), info[1:]))
    return tuner, results


def main(args):
    summary = []
    my_source = f"./{args.exp}_scalapack_mmax{args.mmax}_nmax{args.nmax}_nodes{args.nodes}_ntask{args.ntask}.txt"
    save_path = f"./{args.exp}_scalapack_mmax{args.mmax}_nmax{args.nmax}_nodes{args.nodes}_ntask{args.ntask}.pkl"
    with open(my_source, "r") as f:
        line = f.readline() 
        while line:
            info = line.split()
            if info[0] == 'Task':
                task_id = int(info[1][:-1])
                m = int(info[-2][1:-1])
                n = int(info[-1][:-1])
                info = f.readline().split()
                runs = [int(x) for x in info[3:]]
                line = f.readline()
                results = []
                while line.strip() != "":
                    results.append(get_results_from_line(line))
                    line = f.readline()
                summary.append((task_id, (m, n), runs, results, max(m,n)*min(m,n)**2))
            line = f.readline()
    # print(summary)
    # sort summary wrt nx*ny*nz
    summary = sorted(summary, key=itemgetter(4))
    pickle.dump(summary, open(save_path, "wb"))  
    # print()
    # print(summary)

if __name__ == "__main__":
    main(parse_args())