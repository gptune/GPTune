#! /usr/bin/env python

#time ./plasma_tuners_compare.py -mmax 1000 -nmax 1000 -nodes 1 -cores 64 -machine $HOSTNAME -tuner common -ntr 10 -nts 10 -nruns 10 -niter 10 -jobid 1 > output &

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import math
import numpy as np
import argparse
import functools
import subprocess
import copy
import pickle
import colored_traceback
colored_traceback.add_hook(always=True)
import sys
import os

GPTUNEDIR = os.path.abspath(__file__ + "/../../GPTune/")
sys.path.insert(0, GPTUNEDIR)
TESTDIR = os.path.abspath(__file__ + "./plasma-driver/plasma/test/")

from autotune.space import *
from autotune.problem import *
from autotune.search import *
from gptune import GPTune
from problem import Problem
from data import Data
from options import Options
from computer import Computer
from sample import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter

def objective(point):

    global ROOTDIR
    global args

    niter = args.niter

    command = f'export LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]}:./plasma-driver/plasma/lib;'
    if ('nth' in point):
        command += f'export OMP_NUM_THREADS={point["nth"]};'
#    dpotrf
#    command += f'./plasma-driver/plasma/test/test dpotrf --dim={point["n"]} --nb={",".join(niter*[str(point["nb"])])};'
#    dgeqrf
    command += f'./plasma-driver/plasma/test/test dgeqrf --iter={niter} --dim={point["m"]}x{point["n"]} --nb={point["nb"]} --ib={point["ib"]};'

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)#, check=True, text=True)
    output, errors = p.communicate()
    print(output)
    print(errors)

#    mytime = min([float(output.split()[11 + 9 * i]) for i in range(niter)])
#    print(point, mytime)
#    return [mytime]
    mygflops = - max([float(output.split()[14 + 10 * i]) for i in range(niter)])
    print(point, mygflops)
    return [mygflops]

#times  = pickle.load(open('times_plasma_dpotrf_th_1_n_1_to_1000.pkl', 'rb'))
#gflops = pickle.load(open('gflops_plasma_dpotrf_th_1_n_1_to_1000.pkl', 'rb'))
#
#def saved_objective(point):
#
##    obj = times[(point['n'],point['nb'])]
##    obj = math.log(max(1e-9, times[(point['n'],point['nb'])]))
#    obj = - gflops[(point['n'],point['nb'])]
#    print(point, obj)
#    return obj

def parse_args():

    # Parse command line arguments

    parser = argparse.ArgumentParser()

    # Problem related arguments
    parser.add_argument('-mmax', type=int, default=-1, help='Number of rows')
    parser.add_argument('-nmax', type=int, default=-1, help='Number of columns')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    parser.add_argument('-tuner', type=str, default='common', help='Optimization algorithm (common, opentuner, hpbandster, gp, lcm, dgpiid, dgpcor)')
#    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-ntr', type=int, default=1, help='Number of train tasks')
    parser.add_argument('-nts', type=int, default=1, help='Number of test  tasks')
    parser.add_argument('-nruns', type=int, help='Number of runs per task')
    parser.add_argument('-niter', type=int, help='Number of times a run is repeated (to reduce noise)')
    # Experiment related arguments
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job (0 means interactive execution, not batch)')

    args = parser.parse_args()

    return args

def define_problem(args):

#    nprocmax = nodes*cores - 1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
#    nprocmin = min(nodes, nprocmax - 1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space 

    mmax = args.mmax
    nmax = args.nmax
    nodes = args.nodes
    cores = args.cores

    m   = Integer(1, mmax, transform="normalize", name="m")
    n   = Integer(1, nmax, transform="normalize", name="n")
    nb  = Integer(1, min(mmax, nmax), transform="normalize", name="nb")
    ib  = Integer(1, min(mmax, nmax), transform="normalize", name="ib")
    nth = Integer(1, cores, transform="normalize", name="nth")
#    np = Integer(0, nprocmax, transform="normalize", name="np")
    t = Real(float("-Inf"), float("Inf"), name="t")

#    IS = Space([n])
    IS = Space([m, n, nth])
#    PS = Space([nb])
    PS = Space([nb, ib])
    OS = Space([t])

    cst1 = "nb <= n"
    cst2 = "nb <= m"
    cst3 = "ib <= nb"
#    constraints = {"cst1": cst1}
    constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}

    tuning_problem = TuningProblem(IS, PS, OS, objective, constraints, None)
    problem  = Problem(tuning_problem)

    return (tuning_problem, problem)

def chose_options(args):

    options = Options()

    # Set options

    options['verbose'] = True

    options['sample_class'] = 'SampleOpenTURNS'
    options['sample_algo'] = 'LHS-MDU'
    options['sample_max_iter'] = 20#10**6

    options['model_restarts'] = 1
    options['model_max_iters'] = 2000
    options['model_inducing'] = None
    options['model_layers'] = 2

    options['search_threads'] = 1
    options['search_class'] = 'SearchPyGMO'
    options['search_algo'] = 'pso'
    options['search_udi'] = 'thread_island'
    options['search_pop_size'] = 1000
    options['search_gen'] = 10
    options['search_evolve'] = 10
    options['search_max_iters'] = 100
    options['search_more_samples'] = 1

    # Validate options
#    options.validate(computer=computer)

    return options

def generate_tasks(args, problem, computer, options):

    # Generate train and test tasks

    Ntr = args.ntr
    Nts = args.nts
    NI  = Ntr + Nts

    sampler = eval(f'{options["sample_class"]}()')
    check_constraints = functools.partial(computer.evaluate_constraints, problem, inputs_only = True, kwargs = None) #XXX add kwargs
    I = sampler.sample_inputs(n_samples = NI, IS = problem.IS, check_constraints = check_constraints, **options)
    I = np.array(problem.IS.inverse_transform(I), ndmin=2)
#    Its = [np.array([i, j]) for i in range(100,1001,100) for j in range(100,1001,100)]

    perm = np.random.permutation(NI)
    idx_tr = np.sort(perm[0:Ntr])
    idx_ts = np.sort(perm[Ntr:Ntr + Nts])
#    Itr = I[idx_tr]
#    Its = I[idx_ts]

    return (NI, I, idx_tr, idx_ts)

def main():

    global args

    args = parse_args()

    tuning_problem, problem = define_problem(args)

    computer = Computer(nodes=args.nodes, cores=args.cores, hosts=None)

    options = chose_options(args)

    tasks_filename = 'tasks.pkl'
    if (not os.path.exists(tasks_filename)):
        (NI, I, idx_tr, idx_ts) = generate_tasks(args, problem, computer, options)
        with open(tasks_filename, 'wb') as f:
            pickle.dump((NI, I, idx_tr, idx_ts), f)
    else:
        with open(tasks_filename, 'rb') as f:
            (NI, I, idx_tr, idx_ts) = pickle.load(f)

    Ntr = len(idx_tr)
    Nts = len(idx_ts)
    NS = args.nruns
    NS1 = max(NS//2, 1) 
    machine = args.machine
    tuner = args.tuner

#    os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    # Common data

    common_train_filename = 'common_train.pkl'
    common_test_filename = 'common_test.pkl'
    common_filename = 'common.pkl'
#    os.path.abspath(filename)
    if ((tuner == 'common') or (not os.path.exists(common_filename))):
        # Train
        common_train_data = Data(problem, I=I[idx_tr])
        common_train_gt = GPTune(tuning_problem, computer=computer, data=common_train_data, options=options)
        (common_train_data, common_train_model, common_train_stats) = common_train_gt.MLA1(NS=NS1, NI=0, Igiven=I[idx_tr], NS1=NS1)
        with open(common_train_filename, 'wb') as f:
            pickle.dump((common_train_gt, common_train_data, common_train_model, common_train_stats), f)
        # Test
        common_test_data = Data(problem, I=I[idx_ts])
        common_test_gt = GPTune(tuning_problem, computer=computer, data=common_test_data, options=options)
        (common_test_data, common_test_model, common_test_stats) = common_test_gt.MLA1(NS=NS1, NI=0, Igiven=I[idx_ts], NS1=NS1)
        with open(common_test_filename, 'wb') as f:
            pickle.dump((common_test_gt, common_test_data, common_test_model, common_test_stats), f)
        # Dataset
#        common_data = Data(problem, I=I)
#        common_data.fusion(common_train_data)
#        common_data.fusion(common_test_data)
        common_train_data.fusion(common_test_data)
        common_data = common_train_data
        with open(common_filename, 'wb') as f:
            pickle.dump(common_data, f)
    else:
        with open(common_train_filename, 'rb') as f:
            (common_train_gt, common_train_data, common_train_model, common_train_stats) = pickle.load(f)
        with open(common_test_filename, 'rb') as f:
            (common_test_gt, common_test_data, common_test_model, common_test_stats) = pickle.load(f)
        with open(common_filename, 'rb') as f:
            common_data = pickle.load(f)

    if (tuner == 'opentuner'):

        for tid in range(NI):
            opentuner_filename = f'opentuner_tid_{tid}.pkl'
            if (not os.path.exists(opentuner_filename)):
                (opentuner_data, opentuner_stats) = OpenTuner(T=common_data.I[tid,None], NS=NS, tp=tuning_problem, computer=computer, run_id="OpenTuner", niter=args.niter, technique=None)
                with open(opentuner_filename, 'wb') as f:
                    pickle.dump((opentuner_data, opentuner_stats), f)

    elif (tuner == 'hpbandster'):

        for tid in range(NI):
            hpbandster_filename = f'hpbandster_tid_{tid}.pkl'
            if (not os.path.exists(hpbandster_filename)):
                (hpbandster_data, hpbandster_stats) = HpBandSter(T=common_data.I[tid,None], NS=NS, tp=tuning_problem, computer=computer, run_id="HpBandSter", niter=args.niter)
                with open(hpbandster_filename, 'wb') as f:
                    pickle.dump((hpbandster_data, hpbandster_stats), f)

    elif (tuner == 'gp'):

        options['model_class'] = 'Model_GPy_LCM'
        options['search_strategy'] = "independant_multitask"
#        options['model_max_iters'] = 1000
        for tid in range(NI):
            gp_filename = f'gp_tid_{tid}.pkl'
            if (not os.path.exists(gp_filename)):
                gp_data = Data(problem, I=common_data.I[tid,None], P=[common_data.P[tid]], O=[common_data.O[tid]])
                gp_gt = GPTune(tuning_problem, computer=computer, data=gp_data, options=options)
                (gp_data, gp_model, gp_stats) = gp_gt.MLA(NS=NS, NI=1, Igiven=None, NS1=0)
                with open(gp_filename, 'wb') as f:
                    pickle.dump((gp_gt, gp_data, gp_model, gp_stats), open(gp_filename, 'wb'))

    elif (tuner == 'lcm'):

        lcm_data = copy.deepcopy(common_train_data)
        options['model_class'] = 'Model_GPy_LCM'
#        options['model_sparse'] = True
#        options['model_latent'] = 5
#        options['model_max_iters'] = 1000
#        options['model_class'] = 'Model_LCM'
#        options['distributed_memory_parallelism'] = False
#        options['shared_memory_parallelism'] = True
#        options['model_restart_processes'] = 1
#        options['model_processes'] = 1
#        options['model_restart_threads'] = 8
#        options['model_threads'] = 1
        options['search_strategy'] = 'independant_multitask'
        lcm_gt = GPTune(tuning_problem, computer=computer, data=lcm_data, options=options)
        (lcm_data, lcm_model, lcm_stats) = lcm_gt.MLA(NS=NS, NI=Ntr, Igiven=None, NS1=0)
        lcm_MLA_filename = f'lcm_MLA.pkl'
        with open(lcm_MLA_filename, 'wb') as f:
            pickle.dump((lcm_gt, lcm_data, lcm_model, lcm_stats), f)
        with open(lcm_MLA_filename, 'rb') as f:
           (lcm_gt, lcm_data, lcm_model, lcm_stats) =  pickle.load(f)

        (lcm_aprxopts, lcm_objval, lcm_stats) = lcm_gt.TLA1(I[idx_ts], NS=None)
        lcm_TLA1_filename = f'lcm_TLA1.pkl'
        with open(lcm_TLA1_filename, 'wb') as f:
            pickle.dump((lcm_gt, I[idx_ts], lcm_aprxopts, lcm_objval, lcm_stats), f)

    elif (tuner == 'dgpiid'):

        dgpiid_data = copy.deepcopy(common_train_data)
        options['model_class'] = 'Model_SGHMC_DGP' # 'Model_DGP'
        options['model_max_iters'] = 20000
        options['search_strategy'] = 'independant_multitask'
        options['shared_memory_parallelism'] = True
        options['search_multitask_threads'] = 30
        dgpiid_gt = GPTune(tuning_problem, computer=computer, data=dgpiid_data, options=options)
        (dgpiid_data, dgpiid_model, dgpiid_stats) = dgpiid_gt.MLA(NS=NS, NI=Ntr, Igiven=None, NS1=0)
        dgpiid_MLA_filename = 'dgpiid_MLA.pkl'
        with open(dgpiid_MLA_filename, 'wb') as f: 
            #pickle.dump((dgpiid_gt, dgpiid_data, dgpiid_model, dgpiid_stats), f)
            pickle.dump((dgpiid_gt, dgpiid_data, dgpiid_stats), f)
#        with open(dgpiid_MLA_filename, 'rb') as f: 
#           (dgpiid_gt, dgpiid_data, dgpiid_model, dgpiid_stats) =  pickle.load(f)

        (dgpiid_aprxopts, dgpiid_objval, dgpiid_stats) = dgpiid_gt.TLA1(I[idx_ts], NS=None)
        dgpiid_TLA1_filename = 'dgpiid_TLA1.pkl'
        with open(dgpiid_TLA1_filename, 'wb') as f: 
            pickle.dump((dgpiid_gt, I[idx_ts], dgpiid_aprxopts, dgpiid_objval, dgpiid_stats), f)

#        dgpiid_gt.options['search_pop_size'] = 1000
        (dgpiid_aprxopts, dgpiid_objval, dgpiid_stats) = dgpiid_gt.TLA2(I[idx_ts], dgpiid_model)
        dgpiid_TLA2_filename = 'dgpiid_TLA2.pkl'
        with open(dgpiid_TLA2_filename, 'wb') as f: 
            pickle.dump((dgpiid_gt, I[idx_ts], dgpiid_aprxopts, dgpiid_objval, dgpiid_stats), f)

    elif (tuner == 'dgpcor'):

        dgpcor_data = Data(problem)
        options['model_class'] = 'Model_SGHMC_DGP' # 'Model_DGP'
        options['model_max_iters'] = 20000
        options['search_strategy'] = 'continuous_correlated_multitask'
        options['shared_memory_parallelism'] = True
        #options['search_multitask_threads'] = 30
        options['search_threads'] = 40
        options['model_update_no_train_iters'] = Ntr
        options['search_correlated_multitask_NX'] = NI
        options['search_correlated_multitask_NA'] = NS
#        options['search_threads'] = 1
        N1 = Ntr * NS1        #= args.ntr
        N2 = Ntr * (NS - NS1) #= args.nruns

        dgpcor_phase1_MLA2_filename = 'dgpcor_phase1_MLA2.pkl'
        if (not os.path.exists(dgpcor_phase1_MLA2_filename)):
            dgpcor_phase1_gt = GPTune(tuning_problem, computer=computer, data=dgpcor_data, options=options)
            (dgpcor_phase1_data, dgpcor_phase1_model, dgpcor_phase1_stats) = dgpcor_phase1_gt.MLA2(N1=N1, N2=0)
            with open(dgpcor_phase1_MLA2_filename, 'wb') as f: 
                pickle.dump((dgpcor_phase1_gt, dgpcor_phase1_data, dgpcor_phase1_model, dgpcor_phase1_stats), f)
        else:
            with open(dgpcor_phase1_MLA2_filename, 'rb') as f: 
                (dgpcor_phase1_gt, dgpcor_phase1_data, dgpcor_phase1_model, dgpcor_phase1_stats) = pickle.load(f)

        dgpcor_phase2_gt = GPTune(tuning_problem, computer=computer, data=dgpcor_phase1_data, options=options)
        (dgpcor_phase2_data, dgpcor_phase2_model, dgpcor_phase2_stats) = dgpcor_phase2_gt.MLA2(N1=0, N2=N2)
        dgpcor_phase2_MLA2_filename = 'dgpcor_phase2_MLA2.pkl'
        with open(dgpcor_phase2_MLA2_filename, 'wb') as f: 
            #pickle.dump((dgpcor_phase2_gt, dgpcor_phase2_data, dgpcor_phase2_model, dgpcor_phase2_stats), f)
            pickle.dump((dgpcor_phase2_gt, dgpcor_phase2_data, dgpcor_phase2_stats), f)
#        with open(dgpcor_phase2_MLA2_filename, 'rb') as f: 
#           (dgpcor_phase2_gt, dgpcor_phase2_data, dgpcor_phase2_model, dgpcor_phase2_stats) = pickle.load(f)

        (dgpcor_aprxopts, dgpcor_objval, dgpcor_stats) = dgpcor_phase2_gt.TLA1(I[idx_ts], NS=None)
        dgpcor_TLA1_filename = 'dgpcor_TLA1.pkl'
        with open(dgpcor_TLA1_filename, 'wb') as f: 
            pickle.dump((dgpcor_phase2_gt, I[idx_ts], dgpcor_aprxopts, dgpcor_objval, dgpcor_stats), f)

#        dgpcor_phase2_gt.options['search_pop_size'] = 1000
        (dgpcor_aprxopts, dgpcor_objval, dgpcor_stats) = dgpcor_phase2_gt.TLA2(I[idx_ts], dgpcor_phase2_model)
        dgpcor_TLA2_filename = 'dgpcor_TLA2.pkl'
        with open(dgpcor_TLA2_filename, 'wb') as f: 
            pickle.dump((dgpcor_phase2_gt, I[idx_ts], dgpcor_aprxopts, dgpcor_objval, dgpcor_stats), f)


if __name__ == "__main__":

    main()

