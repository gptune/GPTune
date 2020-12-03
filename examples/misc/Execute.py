#!/usr/bin/env python3
'''
This script is used to provide a uniform interface for GPtune
1. preprocess : generating input config file for C code based on the template config file
2. execute    : running the binary code
3. postprocess: generating the log dir and record the result
'''
import re
import os 
import subprocess
import time
import argparse
import numpy as np
import itertools
import random


def IsNumerical(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class Running:
    def __init__(self,e_path,size,TemplateInput,keys,vals):
        log_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_path = os.path.join(os.getcwd(),'log',log_name)
        count = 1
        while os.path.exists(log_path):
            log_path = log_path+'_'+str(count) 
            count = count + 1
        os.makedirs(log_path)
        
        self.log_path = log_path
        self.batch_size = size 
        self.exec_path = e_path 

        self.config = {}
        self.input = TemplateInput
        self.keys = keys
        self.vals = vals
        self.output_config = os.path.join(self.log_path,'config.dat')

        self.max_it = 1000
        self.max_time = 10000.0
        self.iterations = []
        self.elapse_times = []
        self.average_time = 0
        self.average_iter = 0

    def ParseInputConfig(self):
        with open(self.input,'r') as f:
            contents = f.readlines()

        for line in contents:
            if re.search('%',line):
                line = line.split('%')[0].strip()

            if re.search('=',line):
                items = line.split('=')
                key = items[0].strip()
                val = items[1].strip()
                self.config[key] = val

    def OutputConfig(self):
        for i in range(len(self.keys)):
            self.config[self.keys[i]] = self.vals[i]

        contents = []
        for k,v in self.config.items():
            line = '{:<25} = {:<25} \n'.format(k,v)
            contents.append(line)

        with open(self.output_config,'w') as f:
            f.writelines(contents)

    def BatchExec(self):
        self.ParseInputConfig()
        self.OutputConfig()
        cmd = self.exec_path +' -ini '+self.output_config

        for i in range(self.batch_size):
            try:
                run_output = subprocess.check_output(cmd,shell=True)
            except subprocess.CalledProcessError as error:
                print('\033[31m running fail! :','check the log dir {} \033[0m'.format(self.log_path))
                print(error.output.decode('utf-8'))
                # contents = error.output.decode('utf-8')
                contents = ['Number of iterations = {} with relative residual 0.\n'.format(self.max_it),'AMG_Krylov method totally costs {} seconds\n'.format(self.max_time)]
            else:
                contents = run_output.decode('utf-8')
            finally:
                local_log_path = os.path.join(self.log_path,str(i)+'.log')
                with open(local_log_path,'w') as f:
                    f.write(contents)
                    change_list = ['\n','='*75,'\n']
                    for k in range(len(self.keys)):
                        tmp_line = '{:<25} = {:<25} \n'.format(self.keys[k],self.vals[k])
                        change_list.append(tmp_line)

                    f.writelines(change_list)

                lines = contents.split('\n')
                for line in lines:
                    if re.search('iterations',line):
                        self.iterations.append(eval(line.split()[4]))
                    if re.search('MaxIt',line):
                        self.iterations.append(eval(line.split()[4]))
                    if re.search('totally',line):
                        self.elapse_times.append(eval(line.split()[4]))

    def CollectInfo(self):
        # the target is iteration, if the target is elapsed time, change the if-else condition as follow
        result_len = len(self.iterations)
        if result_len == 0:
            print('\033[31m can not collect the info : the len of result is 0, check the log dir {} \033[0m'.format(self.log_path))
            self.average_iter = self.max_it
            # self.average_time = self.max_time
        else:
            for i in range(result_len):
                if not IsNumerical(self.iterations[i]):
                    print('\033[31m the value is not a number, check the log dir {} \033[0m'.format(self.log_path))
                    self.iterations[i] = self.max_it
                elif self.iterations[i] == 0: 
                    print('\033[31m the value is 0, check the log dir {} \033[0m'.format(self.log_path))
                elif abs( self.iterations[i] - self.iterations[0] ) > 5:
                    print("be careful! the iterations[{}] changes rapidly".format(i))


            # self.average_time = sum(self.elapse_times) / len(self.elapse_times)
            self.average_iter = sum(self.iterations) / result_len

        print("the iteration_num = {}".format(self.average_iter))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, dest='batch', help='how many times the same program runs')
    parser.add_argument('--keys',type=str,action='append', dest='keys', help='the name of parameters')
    parser.add_argument('--vals',type=str,action='append', dest='vals', help='the value of parameters')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()

    # modify the args.keys and args.vals for the class Running
    # change the kappa id into the dir name 
    args.vals[0] = './{}/'.format(args.vals[0])


    # a = Running('/home/zhf/software/fasp/icf/test',args.batch,'/home/zhf/software/fasp/icf/input.dat',args.keys,args.vals)
    a = Running('./test',args.batch,'./input.dat',args.keys,args.vals)
    a.BatchExec()
    a.CollectInfo()



