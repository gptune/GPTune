import json
import argparse
import numpy as np
import pickle
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', type=str,default='unknown',help='name of this run log')
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    args = parser.parse_args()

    return args

def main(args):
    with open("./gptune.db/STRUMPACK_KRR.json") as f:
        data = json.load(f)

    if args.ntask==1:
        tid = 0
        history = []
        for item in data["func_eval"]:
            budget = item['task_parameter']['budget']
            fval = item['evaluation_result']['r']
            history.append([budget, fval])
            task = item['task_parameter']['datafile']
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