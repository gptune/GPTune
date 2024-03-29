import json
import argparse
import numpy as np
import pickle
import re
from itertools import islice

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str,default='r',help='objective name')
    parser.add_argument('-save_path', type=str,default='unknown',help='name of this run log')
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument('-database', type=str, default='unknown', help='path to the database')
    args = parser.parse_args()

    return args

def main(args):
    with open(args.database) as f:
        data = json.load(f)

    if args.ntask==1:
        tid = 0
        history = []
        for item in data["func_eval"]:
            budget = item['task_parameter']['budget']
            fval = item['evaluation_result'][args.r]
            history.append([budget, fval])
            task = item['task_parameter']
            # print(type(task),task,task.items())
            task = [key+':'+str(value) for key, value in islice(task.items(), 1, None)]
            # print(type(task),task)
        x = []
        y = []
        pre_fix = 0
        max_num = -999
        for info in history:
            if info[0] > max_num:
                max_num = info[0]
        for info in history:
            pre_fix += info[0]/max_num
            if np.isclose(info[0], max_num):
                x.append(pre_fix)
                y.append(info[1])
        results = [tid, task, [x,y]]
    print("Finish parseing GPTuneBand results")
    print(results)
    print(f"saved path: {args.save_path}_parsed.pkl")
    pickle.dump(results, open(f"{args.save_path}_parsed.pkl", "wb"))  
    
    
if __name__ == "__main__":
    main(parse_args())