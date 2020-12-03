import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="poisson", help ='type of PDE to solve')
    return parser.parse_args()

def get_results_from_line(line):
    info = line.split("\t")
    tuner = info[0]
    results = list(map(lambda x: float(x.split()[0]), info[1:]))
    return tuner, results


def main(args):
    summary = []
    my_source = f"./exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.txt"
    save_path = f"./exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.pkl"
    with open(my_source, "r") as f:
        line = f.readline() 
        while line:
            info = line.split()
            if info[0] == 'Task':
                print(info)
                task_id = int(info[1][:-1])
                nx = int(info[-3][1:-1])
                ny = int(info[-2][:-1])
                nz = int(info[-1][:-1])
                info = f.readline().split()
                print(info)

                runs = [int(x) for x in info[3:]]
                line = f.readline()
                results = []
                while line.strip() != "":
                    results.append(get_results_from_line(line))
                    line = f.readline()
                summary.append((task_id, (nx, ny, nz), runs, results, nx*ny*nz))
            line = f.readline()
    # print(summary)
    # sort summary wrt nx*ny*nz
    summary = sorted(summary, key=itemgetter(4))
    pickle.dump(summary, open(save_path, "wb"))  
    # print()
    # print(summary)

if __name__ == "__main__":
    main(parse_args())