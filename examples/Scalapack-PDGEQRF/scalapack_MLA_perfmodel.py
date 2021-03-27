#! /usr/bin/env python3

"""
Example of invocation of this script:

mpirun -n 1 python scalapack_MLA_perfmodel.py -mmax 5000 -nmax 5000 -nprocmin_pernode 1 -ntask 20 -nrun 800 -nrun1 400 -perfmodel 1 -jobid 0

where:
    -mmax (nmax) is the maximum number of rows (columns) in a matrix
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task 
    -nrun1 is the number of initial samples per task 
    -perfmodel is whether a coarse performance model is used
    -jobid is optional. You can always set it to 0.
"""

################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../scalapack-driver/spt/"))

from pdqrdriver import pdqrdriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import numpy as np
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import time
from sklearn.linear_model import LinearRegression
import math


################################################################################

''' The objective function required by GPTune. '''

# 
def print_model_predication(point):

######################################### 
##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']
    bunit = point['bunit']	
#########################################
    
    m = point['m']
    n = point['n']
    p = point['p']
    mb = point['b']*bunit
    nb = point['b']*bunit
    npernode = 2**point['npernode']
    nproc = nodes*npernode
    nthreads = int(cores / npernode)  

    q = int(nproc / p)
    nproc = p*q
    b = max(mb,nb)
    
    flops = (2*n**2*(3*m-n)/3/nproc + b*n**2/2/q + 3*b*n*(2*m-n)/2/p + b**2*n/3/p  )/(max(m,n)*min(m,n)**2) 
    divides = (m*n-n**2/2)/p/min(m,n)**2 *4  # Assuming 1 divides uses 4 flops
    messages = (3*n*np.log2(p)+2*n/b*np.log2(q))/min(m,n)
    words = ((n**2/q+b*n)*np.log2(p) +((m*n-n**2/2)/p+b*n/2)*np.log2(q))/min(m,n)**2 


    ''' assuming only 80% of the peak Gflops (36.8), use bandwidith based on the average message size (words/messages) and a ping-pong benchmark'''
    # measured = [1/(36.8*1e9)*(max(m,n)*min(m,n)**2)/0.8,1/(36.8*1e9)*(min(m,n)**2)/0.8, 11.47*1e-6*min(m,n), min(m,n)**2*8/90.287/1e6]
    measured = [1/(36.8*1e9)*(max(m,n)*min(m,n)**2)/0.8,1/(36.8*1e9)*(min(m,n)**2)/0.8, 11.47*1e-6*min(m,n), min(m,n)**2*8/90.287/1e6,  words*min(m,n)**2*8/(n/b), words*min(m,n)*8/(messages)]
    print('measured coefficients',measured)
    print('model1(data fit):','flop term:', "{:.2e}".format(flops*(point['c0'])),'divide term:', "{:.2e}".format(divides*(point['c1'])),'latency term:',"{:.2e}".format(messages*(point['c2'])),'volume term:',"{:.2e}".format(words*(point['c3'])),'offset term:',"{:.2e}".format(words*(point['c4'])),'total:',"{:.2e}".format(flops*(point['c0'])+divides*(point['c1'])+ messages*(point['c2'])+words*(point['c3'])+words*(point['c4'])))
    print('model2(measured):','flop term:', "{:.2e}".format(flops*(measured[0])),'divide term:', "{:.2e}".format(divides*(measured[1])),'latency term:',"{:.2e}".format(messages*(measured[2])),'volume term:',"{:.2e}".format(words*(measured[3])),'total:',"{:.2e}".format(flops*(measured[0])+divides*(measured[1])+messages*(measured[2])+words*(measured[3])))


# should always use this name for user-defined model function
def models(point):

######################################### 
##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']
    bunit = point['bunit']	
#########################################


    m = point['m']
    n = point['n']
    p = point['p']
    bunit = point['bunit']
    nodes = point['nodes']
    mb = point['b']*bunit
    nb = point['b']*bunit
    npernode = 2**point['npernode']
    nproc = nodes*npernode


    # print('testing: ',point['c0'],point['c1'])
    # this becomes useful when the parameters returned by TLA1 do not respect the constraints
    if(nproc == 0 or p == 0 or nproc < p):
        print('Warning: wrong parameters for models function!!!', nproc,p)
        return 1e12

    q = int(nproc / p)
    nproc = p*q

    b = max(mb,nb)
    
    
    flops = (2*n**2*(3*m-n)/3/nproc + b*n**2/2/q + 3*b*n*(2*m-n)/2/p + b**2*n/3/p  )/(max(m,n)*min(m,n)**2) 
    divides = (m*n-n**2/2)/p/min(m,n)**2 *4  # 1 divides uses 4 flops
    messages = (3*n*np.log2(p)+2*n/b*np.log2(q))/min(m,n)
    words = ((n**2/q+b*n)*np.log2(p) +((m*n-n**2/2)/p+b*n/2)*np.log2(q))/min(m,n)**2 

    return [flops*(point['c0'])+divides*(point['c1'])+ messages*(point['c2'])+words*(point['c3'])+(point['c4'])]

    # flops = max(m,n)*min(m,n)**2/nproc/(max(m,n)*min(m,n)**2)
    # return [flops]

def models_update(data):


    # Xall = np.empty(shape=[0, n])
    # Yall = 
    for i in range(len(data.I)):
        X0=np.array(data.P[i])
        y=np.array(data.O[i])        
        
        # X0=np.array([[3, 462, 185], [8, 264, 47], [2, 306, 224], [6, 273, 88], [5, 408, 124], [1, 381, 262], [2, 494, 55], [12, 317, 22], [2, 292, 136], [5, 357, 28], [7, 398, 6], [2, 451, 238], [3, 483, 85], [7, 468, 35], [5, 393, 67], [2, 462, 148], [6, 303, 43], [3, 487, 205], [1, 346, 15], [6, 391, 87], [4, 455, 50], [10, 286, 13], [12, 290, 30], [3, 486, 14], [2, 372, 183], [1, 305, 149], [13, 271, 34], [10, 277, 51], [4, 418, 144], [14, 476, 21], [3, 509, 126], [10, 383, 62], [1, 355, 74], [1, 472, 292], [11, 309, 38], [2, 334, 230], [6, 364, 101], [9, 454, 45], [1, 365, 356], [7, 418, 20], [4, 312, 4], [3, 467, 4], [1, 496, 5], [2, 505, 4], [2, 363, 3], [2, 304, 1], [8, 505, 7], [2, 295, 2], [16, 508, 14], [16, 306, 15], [11, 261, 7], [3, 507, 17], [1, 478, 6], [2, 481, 4], [3, 441, 6], [3, 315, 2], [5, 259, 8], [1, 482, 5], [1, 508, 467], [1, 421, 7], [12, 503, 48], [15, 375, 17], [13, 342, 10], [1, 416, 5], [1, 446, 9], [3, 458, 5], [3, 484, 6], [6, 271, 3], [3, 335, 2], [1, 431, 8], [3, 416, 3], [1, 481, 7], [2, 376, 4], [6, 476, 13], [3, 495, 5], [2, 371, 2], [5, 508, 6], [1, 323, 2], [2, 429, 3], [2, 361, 2]])
        # y=np.array([[1.53719], [0.768771], [2.43541], [1.010937], [1.058509], [2.833908], [0.550157], [0.404064], [1.310177], [0.423595], [0.325993], [2.410096], [0.83457], [0.4903], [0.711798], [1.327985], [0.61927], [1.601474], [0.312268], [0.909908], [0.532612], [0.345859], [0.522319], [0.294955], [1.714039], [1.62173], [0.605689], [0.796724], [1.405523], [0.438243], [1.005741], [0.72856], [0.909214], [3.293102], [0.632621], [2.491231], [1.049698], [0.569117], [3.383905], [0.415036], [0.246633], [0.217687], [0.187744], [0.194066], [0.197286], [0.337386], [0.26556], [0.231152], [0.407284], [0.397056], [0.343938], [0.342542], [0.18816], [0.189801], [0.202527], [0.262211], [0.257169], [0.188608], [3.347614], [0.202936], [0.517257], [0.37979], [0.36881], [0.223301], [0.214468], [0.208882], [0.202667], [0.308692], [0.260794], [0.204905], [0.225971], [0.194198], [0.204637], [0.286863], [0.208304], [0.227331], [0.238497], [0.218949], [0.197642], [0.227869]])



        Ns = X0.shape[0]

        b = X0[np.ix_(np.linspace(0,Ns-1,Ns,dtype=int),[0])]*data.D[i]['bunit']

        npernode = X0[np.ix_(np.linspace(0,Ns-1,Ns,dtype=int),[1])]
        npernode = 2**npernode
        nproc = data.D[i]['nodes']*npernode
        p = X0[np.ix_(np.linspace(0,Ns-1,Ns,dtype=int),[2])]
        q = np.floor(nproc/p)
        m = data.I[i][0]
        n = data.I[i][1]
        flops = (2*n**2*(3*m-n)/3/nproc + b*n**2/2/q + 3*b*n*(2*m-n)/2/p + b**2*n/3/p  )/(max(m,n)*min(m,n)**2)
        divides = (m*n-n**2/2)/p/min(m,n)**2 *4  # Assuming 1 divides uses 4 flops
        messages = (3*n*np.log2(p)+2*n/b*np.log2(q))/min(m,n)
        words = ((n**2/q+b*n)*np.log2(p) +((m*n-n**2/2)/p+b*n/2)*np.log2(q))/min(m,n)**2 


        # X = np.hstack((flops,divides,words))
        # X = np.hstack((flops,divides, messages,words))
        X = np.hstack((messages,words))
        # X = np.hstack((flops, messages,words))
        reg = LinearRegression(fit_intercept=False,normalize=False).fit(X, y)
        # print(fitting score: reg.score(X, y))
        # data.D[i]['c0']=reg.coef_[0][0]
        # data.D[i]['c1']=reg.coef_[0][1]
        data.D[i]['c2']=reg.coef_[0][0]
        data.D[i]['c3']=reg.coef_[0][1]
        # data.D[i]['c4']=reg.intercept_[0]
 
        print('task ', i, ': models update: ', data.D[i])



def objectives(point):
######################################### 
##### constants defined in TuningProblem
    nodes = point['nodes']
    cores = point['cores']
    bunit = point['bunit']	
    perfmodel = point['perfmodel']    
#########################################    

    m = point['m']
    n = point['n']
    mb = point['b']*bunit
    nb = point['b']*bunit
    p = point['p']
    npernode = 2**point['npernode']
    nproc = nodes*npernode
    nthreads = int(cores / npernode)    

    # this becomes useful when the parameters returned by TLA1 do not respect the constraints
    if(nproc == 0 or p == 0 or nproc < p):
        print('Warning: wrong parameters for objective function!!!')
        return 1e12
    q = int(nproc / p)
    nproc = p*q
    params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]

    if(perfmodel==1):
        print_model_predication(point)

    elapsedtime = pdqrdriver(params, niter=3, JOBID=JOBID)
    print(params, ' scalapack time: ', elapsedtime)

    return elapsedtime

def cst1(b,p,m,bunit):
    return b*bunit * p <= m
def cst2(b,npernode,n,p,nodes,bunit):
    return b * bunit * nodes * 2**npernode <= n * p
def cst3(npernode,p,nodes):
    return nodes * 2**npernode >= p

def main():

    global nodes
    global cores
    global bunit
    global JOBID

    # Parse command line arguments
    args = parse_args()

    mmax = args.mmax
    nmax = args.nmax
    ntask = args.ntask
    nprocmin_pernode = args.nprocmin_pernode
    nrun = args.nrun
    nrun1 = args.nrun1
    if(nrun1 is None):
        nrun1=max(nrun//2, 1)
    truns = args.truns
    JOBID = args.jobid
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel
    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    os.system("mkdir -p scalapack-driver/bin/%s; cp ../../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores

    bunit=8     # the block size is multiple of bunit
    mmin=1280
    nmin=1280

    m = Integer(mmin, mmax, transform="normalize", name="m")
    n = Integer(nmin, nmax, transform="normalize", name="n")
    b = Integer(4, 16, transform="normalize", name="b")
    npernode     = Integer     (int(math.log2(nprocmin_pernode)), int(math.log2(cores)), transform="normalize", name="npernode")
    p = Integer(1, nprocmax, transform="normalize", name="p")
    r = Real(float("-Inf"), float("Inf"), name="r")

    IS = Space([m, n])
    PS = Space([b, npernode, p])
    OS = Space([r])

    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    constants={"nodes":nodes,"cores":cores,"bunit":bunit,"perfmodel":perfmodel}
    print(IS, PS, OS, constraints)
    if(perfmodel==1):
        problem = TuningProblem(IS, PS, OS, objectives, constraints, models, constants=constants) # use performance models
    else:
        problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    """ Set and validate options """
    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    # options['search_multitask_processes'] = 1
    # options['model_restart_processes'] = 1
    # options['model_restart_threads'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class'] = 'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)

    seed(1)
    if ntask == 1:
        giventask = [[mmax,nmax]]
    else:
        giventask = [[randint(mmin,mmax),randint(nmin,nmax)] for i in range(ntask)]
    ntask=len(giventask)
    data = Data(problem,D=[{'bunit':bunit,'nodes':nodes, 'c0': 0, 'c1': 0,'c2': 0,'c3': 0,'c4': 0}]*ntask)

    # # giventask = [[177, 1303],[367, 381],[1990, 1850],[1123, 1046],[200, 143],[788, 1133],[286, 1673],[1430, 512],[1419, 1320],[622, 263] ]

    # # the following will use only task lists stored in the pickle file
    # data = Data(problem,D=[{'c0': 0, 'c1': 0,'c2': 0,'c3': 0,'c4': 0}]*len(giventask))

    if(TUNER_NAME=='GPTune'):
        if(perfmodel==1):
            gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__),models_update=models_update)
        else: 
            gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__),models_update=None)
        """ Building MLA with NI random tasks """
        NI = ntask
        NS = nrun
        NS1 = nrun1
        (data, model, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS1)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='opentuner'):
        NI = ntask
        NS = nrun
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = ntask
        NS = nrun
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    m:%d n:%d" % (data.I[tid][0], data.I[tid][1]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

def parse_args():

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, help='Number of runs per task')
    parser.add_argument('-nrun1', type=int, help='Number of intial runs per task')
    parser.add_argument('-truns', type=int, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use a performance model')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
