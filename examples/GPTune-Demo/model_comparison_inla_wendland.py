#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


"""
Comparison of two sparse GP models of george with SuperLU_DIST, trained by GPTune (Model_George) with four
optimizers, a copy of model_comparison_updated_superlu.py:
  wendland  the compactly supported Wendland C2 kernel (sparse covariance matrix, model_kern='WendlandC2')
  inla      the INLA/SPDE Matern model on a lattice (sparse precision matrix, model_kern='INLA', george/inla.py),
            2D and 3D objectives only
The environment selects the runs: GP_NS (sample counts, default 102400), GP_MODELS (default "wendland,inla"),
GP_OPTIMIZERS (default "gradient,finite difference,mcmc,mala": the L-BFGS trainings first, they set the time
limit of the samplers), GP_INLA_SHAPE (lattice nodes per dimension, default sqrt(N)) and GP_INLA_NU (the INLA
Matern smoothness, default 2 - d/2). The objective is
read from the standard input (4: 2D Schwefel). The files of every model go to GP_OUTPUT_DIR/<model> (default
./<model>): the ones of model_comparison_updated_superlu.py and its plots (plot_training_results.py), plus
the comparison of the models at every N, model_compare_obj<o>_N<N>.pdf/png/csv in GP_OUTPUT_DIR.
Run with the SuperLU_DIST workers by run_inla_wendland_gpu.sbatch.
"""


################################################################################
import sys
import os
# GPTune and george (for its INLA model, george/inla.py) of the source trees of this repository, which the
# installed packages may predate
GPTUNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, GPTUNE_ROOT)
import george
george.__path__.insert(0, os.path.join(GPTUNE_ROOT, 'george/src/george'))
# import mpi4py
import logging
import json
from pdbridge import *
# george's sparse solver (model_sparse) calls superlu_factor/logdet/solve, but george/solvers/basic.py
# no longer imports pdbridge since the ButterflyPACK solver was added, so provide them here
import george.solvers.basic
for _f in (superlu_factor, superlu_logdet, superlu_solve):
    setattr(george.solvers.basic, _f.__name__, _f)


logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from GPTune.gptune import * # import all
from mpl_toolkits.mplot3d import Axes3D


import argparse
# from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
from plot_training_results import plot_training_histories, plot_optimizer_scaling

from GPTune.callopentuner import OpenTuner
from GPTune.callhpbandster import HpBandSter



# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')

    args = parser.parse_args()

    return args

def objectives1(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1

    # print('test:',test)
    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f]

def objectives2(point):
    x1 = point["x1"]
    x2 = point["x2"]
    y = 4*(x1**2) + 4*(x2**2)
    return [y]

def objectives3(point):
    x1 = point["x1"]
    x2 = point["x2"]
    x3 = point["x3"]
    x4 = point["x4"]
    x5 = point["x5"]
    x6 = point["x6"]
    y = -1*(25*((x1-2)**2) + (x2-2)**2 + (x3-1)**2 + (x4-4)**2 + (x5-1)**2 + (x6-1)**2)
    return [y]
    
def schwefel2_raw(x1, x2):
   
    z1 = -500.0 + 1000.0 * x1
    z2 = -500.0 + 1000.0 * x2

    d = 2
    y = 418.9829 * d - (
        z1 * np.sin(np.sqrt(abs(z1))) +
        z2 * np.sin(np.sqrt(abs(z2)))
    )
    return y

def objectives4(point):
    x1 = point["x1"]
    x2 = point["x2"]

    y_raw = schwefel2_raw(x1, x2)
    y = y_raw / (418.9829 * 2) - 1.0

    return [y]


def predict_aug(modeler, gt, point,tid,objtype):   # point is the orginal space

    if(objtype==1):
        x =point['x']
        x = [x]
    elif(objtype==2):
        x1 =point['x1']
        x2 =point['x2']
        x = [x1,x2]  
    elif(objtype==3):
        x1 =point['x1']
        x2 =point['x2']          
        x3 =point['x3']
        x4 =point['x4']  
        x5 =point['x5']
        x6 =point['x6']  
        x = [x1,x2,x3,x4,x5,x6]
    elif(objtype==4):
        x1 =point['x1']
        x2 =point['x2']
        x = [x1,x2]

    # x =point['x']
    xNorm = gt.problem.PS.transform([x])
    xi0 = gt.problem.PS.inverse_transform(np.array(xNorm, ndmin=2))
    xi=xi0[0]

    IOrig = gt.data.I[tid]

    # point0 = gt.data.D
    point2 = {gt.problem.IS[k].name: IOrig[k] for k in range(gt.problem.DI)}
    point  = {gt.problem.PS[k].name: xi[k] for k in range(gt.problem.DP)}
    # point.update(point0)
    point.update(point2)
    # print("point", point)

    xNorm = gt.problem.PS.transform(xi0)[0]
    if(gt.problem.models is not None):
        if(gt.problem.driverabspath is not None):
            modulename = Path(gt.problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(gt.problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module
        else:
            raise Exception('performance models require passing driverabspath to GPTune')
        # modeldata= self.problem.models(point)
        modeldata= module.models(point)
        xNorm = np.hstack((xNorm,modeldata))  # YL: here tmpdata in the normalized space, but modeldata is the in the original space
        # print(xNorm)
    (mu, var) = modeler[0].predict(xNorm, tid=tid)
    return (mu, var)


NTEST = 1000 # held-out test points (not used for training) for the RMSE and CRPS of the GP along its training
# the compared models: their george kernel and the name of the GP model in the figures
MODELS = {'wendland': ('WendlandC2', 'Sparse GP, Wendland C2 (george + SuperLU)'),
          'inla': ('INLA', 'INLA/SPDE GP (george + SuperLU)')}
OUTPUT_DIR = os.environ.get('GP_OUTPUT_DIR', '.')
import plot_training_results


def model_dir(model_name):
    """The directory of the result files of a model."""
    directory = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(directory, exist_ok=True)
    return directory
REPLAY_POINTS = 20 # history entries replayed for the test metrics of a training (evenly spaced, including the last)


def training_test_metrics(modeler, gt, obj_func, ntest, seed=2026, batch=250, max_replay=REPLAY_POINTS):
    """
    Held-out accuracy of a george GP along its training: for the points recorded in
    modeler.train_history (L-BFGS iterations, or improvements of the best MCMC sample), refactor the
    covariance matrix at those hyperparameters and predict ntest uniformly random points that are not
    used for training. Returns the history entries extended with the RMSE of the predictive mean and
    the mean CRPS of the Gaussian predictive distributions, and the test points and predictions. At
    most max_replay entries, evenly spaced in the history and including the last one, are replayed;
    the others get NaN. Predictions are done in batches, each needing one multi-RHS solve.
    """
    from scipy.stats import norm
    PS = gt.problem.PS
    rng = np.random.default_rng(seed)
    xtest = [[rng.uniform(d.low, d.high) for d in PS.dimensions] for _ in range(ntest)]
    ytest = np.array([obj_func(dict(zip(PS.dimension_names, p)))[0] for p in xtest])
    xtest_norm = np.array(PS.transform(xtest))
    y = np.ravel(modeler.y)
    p_final = modeler.M.get_parameter_vector()
    history = modeler.train_history
    replay = set(np.round(np.linspace(0, len(history) - 1, min(max_replay, len(history)))).astype(int))
    metrics = {} # hyperparameters -> (RMSE, CRPS)
    predictions = {'xtest': np.array(xtest), 'ytest': ytest, 'history_entry': [], 'time': [], 'hyperparameters': [], 'mu': [], 'var': []}
    rows = []
    for i, entry in enumerate(history):
        key = tuple(entry['hyperparameters'])
        if i in replay and key not in metrics:
            start = time.time()
            modeler.M.set_parameter_vector(entry['hyperparameters'])
            mu = np.empty(ntest)
            var = np.empty(ntest)
            for s in range(0, ntest, batch):
                mu[s:s+batch], var[s:s+batch] = modeler.M.predict(y, xtest_norm[s:s+batch], return_var=True)
            sd = np.sqrt(np.maximum(var, 1e-300))
            z = (ytest - mu) / sd
            crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
            metrics[key] = (float(np.sqrt(np.mean((mu - ytest)**2))), float(np.mean(crps)))
            for name, value in (('history_entry', i), ('time', entry['time']), ('hyperparameters', entry['hyperparameters']), ('mu', mu), ('var', var)):
                predictions[name].append(value)
            print("test metrics at history entry %d (training time %.2f s): -loglikelihood %.8e, RMSE %.6e, CRPS %.6e (replay %.1f s)"
                  % (entry['iteration'], entry['time'], entry['nll'], metrics[key][0], metrics[key][1], time.time() - start))
        rmse, crps = metrics.get(key, (np.nan, np.nan))
        rows.append(dict(entry, rmse=rmse, crps=crps))
    modeler.M.set_parameter_vector(p_final)
    return rows, {name: np.array(value) for name, value in predictions.items()}


def write_training_metrics(rows, filename):
    with open(filename, 'w') as f:
        f.write('iteration,time,nfev,ngev,nll,rmse,crps,' + ','.join('hp_%d' % i for i in range(len(rows[0]['hyperparameters']))) + '\n')
        for r in rows:
            f.write('%d,%.6f,%d,%d,%.12e,%.12e,%.12e,' % (r['iteration'], r['time'], r['nfev'], r['ngev'], r['nll'], r['rmse'], r['crps'])
                    + ','.join('%.12e' % v for v in r['hyperparameters']) + '\n')


def write_run_stats(filename, **fields):
    """The settings and GPTune statistics of a run, for plots made later (plot_training_results.py)."""
    import json
    with open(filename, 'w') as f:
        json.dump(fields, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def model_runtime(model, obj_func, NS_input,objtype,optimizer,plotgp,modelhodlr=False, modelsparse=False, model_name='wendland'):
    output_dir = model_dir(model_name)
    import matplotlib
    matplotlib.use('Agg')    
    import matplotlib.pyplot as plt
    global nodes
    global cores


    # Parse command line arguments
    args = parse_args()
    ntask = args.ntask
    nrun = NS_input
    tvalue = args.tvalue
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("\nmachine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    # parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    
    x = Real(0., 1., transform="normalize", name="x")
    x1 = Real(0., 1., transform="normalize", name="x1")
    x2 = Real(0., 1., transform="normalize", name="x2")
    x3 = Real(0., 1., transform="normalize", name="x3")
    x4 = Real(0., 1., transform="normalize", name="x4")
    x5 = Real(0., 1., transform="normalize", name="x5")
    x6 = Real(0., 1., transform="normalize", name="x6")    
    if(objtype==1):
        parameter_space = Space([x])    
    elif(objtype==2):
        parameter_space = Space([x1,x2])    
    elif(objtype==3):
        parameter_space = Space([x1,x2,x3,x4,x5,x6])
    elif(objtype==4):
        parameter_space = Space([x1,x2])

    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    #constraints = {"cst1": "x >= 0. and x <= 1."}
    constraints = {}
    if(perfmodel==1):
        problem =  (input_space, parameter_space,output_space, obj_func, constraints, models)  # with performance model
    else:
        problem = TuningProblem(input_space, parameter_space,output_space, obj_func, constraints, None)  # no performance model

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    # options['objective_evaluation_parallelism'] = True
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 4
    # options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    options['model_optimzier'] = 'lbfgs'

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    # Use the following two lines if you want to specify a certain random seed for the random pilot sampling
    # options['sample_algo'] = 'MCS'
    options['sample_class'] = 'SampleOpenTURNS' #'SampleLHSMDU' 
    options['sample_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for surrogate modeling
    options['model_class'] = model #'Model_George_LCM'#'Model_George_LCM'  #'Model_LCM'
    options['model_kern'] = 'RBF' #'Matern32' #'RBF' #'Matern52'
    if(modelhodlr==True):
        options['model_hodlr'] = True
        options['model_hodlrleaf'] = 200
        options['model_hodlrtol'] = 1e-10
        options['model_hodlrtol_abs'] = 1e-20
        options['model_hodlr_sym'] = 0
        options['model_hodlr_knn'] = 0
        options['model_jitter'] = 0 # 1e-5 # 1e-3

    if(modelsparse==True):
        options['model_sparse'] = True
        options['model_kern'] = 'WendlandC2'

        # about 78 neighbours per point: the cutoff shrinks like 1/sqrt(N), 0.0158 at N=100k (0.005 at N=1M)
        options['model_cutoff'] = 0.0158 * np.sqrt(100000 / (NS_input - 1))

        print("SPARSE CONFIG: N=%d cutoff=%g" % 
              (NS_input - 1,options['model_cutoff']))
        if MODELS[model_name][0] == 'INLA':
            options['model_kern'] = 'INLA'
            # about one lattice node per sample by default
            options['model_inla_shape'] = int(os.environ.get('GP_INLA_SHAPE', int(round(np.sqrt(NS_input - 1)))))
            if os.environ.get('GP_INLA_NU'):
                options['model_inla_nu'] = float(os.environ['GP_INLA_NU'])  # the Matern smoothness (default 2 - d/2)
            print("INLA CONFIG: N=%d lattice of %d nodes per dimension" % (NS_input - 1, options['model_inla_shape']))
    
    # Temporary hardcode 
    if(optimizer == "gradient"):
        options['model_mcmc'] = False
        options['model_grad'] = True
        options['model_grad_nprobe'] = 64 # random probe vectors for the trace terms of the sparse gradient
    elif (optimizer == "mcmc"):
        options['model_mcmc'] = True
        options['model_grad'] = False
        options['model_mcmc_sampler'] = 'MetropolisHastings' # 'Ensemble_emcee', 'MetropolisHastings'
        options['model_mcmc_nchain'] = 2
        options['model_mcmc_burnin'] = 100
        options['model_mcmc_maxiter'] = 500
    elif (optimizer == "mala"):
        options['model_mcmc'] = True
        options['model_grad'] = True # MALA needs the gradient of the log-likelihood
        options['model_grad_nprobe'] = 64 # random probe vectors for the traces of the gradient and the Fisher information
        options['model_mcmc_sampler'] = 'MALA'
        options['model_mcmc_nchain'] = 2
        options['model_mcmc_burnin'] = 20
        options['model_mcmc_maxiter'] = 150
        options['model_mala_fisher_interval'] = 1 # compute the Fisher information preconditioner at every proposal (exact sampling)
        options['model_mala_step_size'] = 1.0
    elif (optimizer == "finite difference"):
        options['model_mcmc'] = False
        options['model_grad'] = False
    else:
        pass
    run_tag = 'obj%d_N%d_%s' % (objtype, NS_input - 1, optimizer.replace(' ', '_'))
    options['model_history_file'] = os.path.join(output_dir, 'training_iterations_%s.csv' % run_tag) # written during the training (see GPTune.model.TrainingHistory)
    if optimizer in ("mcmc", "mala"):
        # if both L-BFGS trainings at this N have been run, the sampling stops (instead of after
        # model_mcmc_maxiter steps) once it has run for 3 times the shorter of their training times
        lbfgs_files = [os.path.join(output_dir, 'training_iterations_obj%d_N%d_%s.csv' % (objtype, NS_input - 1, o)) for o in ('gradient', 'finite_difference')]
        if all(os.path.exists(f) for f in lbfgs_files):
            lbfgs_times = [np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True))['time'][-1] for f in lbfgs_files]
            options['model_mcmc_max_time'] = 3 * min(lbfgs_times)
            options['model_mcmc_maxiter'] = 1000000
            print("MCMC sampling time limit: %.1f s (3 x the shorter L-BFGS training, %s)" % (options['model_mcmc_max_time'], lbfgs_times))

    options['model_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for the search phase
    # options['search_class'] = 'SearchSciPy'
    options['search_random_seed'] = 0

    #options['search_class'] = 'SearchSciPy'
    #options['search_algo'] = 'l-bfgs-b'

    options['search_more_samples'] = 1
    options['search_af']='EI'
    options['search_pop_size']=100
    options['search_gen']=2
    # options['search_ucb_beta']=0.01


    options['verbose'] = True
    options['debug'] = False
    options.validate(computer=computer)

    # print(options)

    if ntask == 1:
        giventask = [[round(tvalue,1)]]
    elif ntask == 2:
        giventask = [[round(tvalue,1)],[round(tvalue*2.0,1)]]
    else:
        giventask = [[round(tvalue*float(i+1),1)] for i in range(ntask)]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']
    if(TUNER_NAME=='GPTune'):
        data = Data(problem)

        # gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__)) # run GPTune with database 
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__), historydb=False) ## Run GPTune without database

        # (data, modeler, stats) = gt.MLA(NS=NS_input, NS1= int(NS_input/2.0), NI=NI, Tgiven=giventask)
        (data, modeler, stats) = gt.MLA(NS=NS_input, NS1= int(NS_input - 1), NI=NI, Tgiven=giventask)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            # print("    Ps ", data.P[tid])
            # print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if len(getattr(modeler[0], 'train_history', [])) > 0:
            rows, predictions = training_test_metrics(modeler[0], gt, obj_func, NTEST, batch=250 if NS_input <= 400001 else 100)
            write_training_metrics(rows, os.path.join(output_dir, 'training_metrics_%s.csv' % run_tag))
            np.savez(os.path.join(output_dir, 'test_predictions_%s.npz' % run_tag), **predictions)
            if hasattr(modeler[0], 'mcmc_chains'):
                np.savez(os.path.join(output_dir, 'mcmc_chains_%s.npz' % run_tag), chains=modeler[0].mcmc_chains, log_posteriors=modeler[0].mcmc_log_posteriors)
            write_run_stats(os.path.join(output_dir, 'run_stats_%s.json' % run_tag), objective=objtype, N=NS_input - 1, optimizer=optimizer, kernel=options['model_kern'],
                            cutoff=options['model_cutoff'], mcmc_max_time=options['model_mcmc_max_time'] if optimizer in ("mcmc", "mala") else None,
                            slurm_job_id=os.environ.get('SLURM_JOB_ID'), date=time.strftime('%Y-%m-%d %H:%M:%S'), stats=stats,
                            final_hyperparameters=modeler[0].M.get_parameter_vector(), final_nll=rows[-1]['nll'],
                            popt=data.P[0][np.argmin(data.O[0])], oopt=float(min(data.O[0])[0]))


        if objtype==1 and plotgp==True:
            # fig = plt.figure(figsize=[12.8, 9.6])
            for tid in range(len(data.I)):
                x = np.arange(0., 1., 0.01)
                fig = plt.figure(figsize=[12.8, 9.6])
                p = data.I[tid]
                t = p[0]
                I_orig=p
                kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}
                y=np.zeros([len(x),1])
                y_mean=np.zeros([len(x)])
                y_std=np.zeros([len(x)])
                for i in range(len(x)):
                    P_orig=[x[i]]
                    kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                    kwargs.update(kwargst)
                    y[i]=objectives1(kwargs)
                    if(TUNER_NAME=='GPTune'):
                        (y_mean[i],var) = predict_aug(modeler, gt, kwargs,tid,objtype)
                        y_std[i]=np.sqrt(var)
                        # print(y_mean[i],y_std[i],y[i])
                fontsize=40
                plt.rcParams.update({'font.size': 40})
                plt.plot(x, y, 'b',lw=2,label='true')

                plt.plot(x, y_mean, 'k', lw=3, zorder=9, label='prediction')
                plt.fill_between(x, y_mean - y_std, y_mean + y_std,alpha=0.2, color='k')
                # print(data.P[tid])
                plt.scatter(data.P[tid], data.O[tid], c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label='sample')

                plt.xlabel('x',fontsize=fontsize+2)
                plt.ylabel('y(t,x)',fontsize=fontsize+2)
                plt.title('t=%f'%t,fontsize=fontsize+2)
                print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())
                # legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
                legend = plt.legend(loc='upper right', shadow=False, fontsize=fontsize)
                # annot_min(x,y)
                # plt.show()
                plt.show(block=False)
                plt.pause(0.5)
                # input("Press [enter] to continue.")
                if(modelhodlr==True):
                    fig.savefig('obj_%s_N_%s_tol_%s.pdf'%(optimizer,int(NS_input - 1),options['model_hodlrtol']))
                elif(modelsparse==True):    
                    fig.savefig('obj_%s_N_%s_superlu.pdf'%(optimizer,int(NS_input - 1)))
                else:
                    fig.savefig('obj_%s_N_%s.pdf'%(optimizer,int(NS_input - 1)))
                    



        if objtype==2 and plotgp==True:
            # fig = plt.figure(figsize=[12.8, 9.6])
            for tid in range(len(data.I)):
                res = 80
                x1 = np.linspace(0., 1., res)
                x2 = np.linspace(0., 1., res)
                X1, X2 = np.meshgrid(x1, x2)

                Y_mean = np.zeros_like(X1)
                Y_std  = np.zeros_like(X1)
                Y_true  = np.zeros_like(X1)

                # kwargst from GPTune.data.I[tid]
                p = data.I[tid]
                t = p[0]
                kwargst = {input_space[k].name: p[k] for k in range(len(input_space))}

                # ====== 2. Evaluate GP ======
                for i in range(res):
                    for j in range(res):
                        P_orig = [x1[i], x2[j]]
                        kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                        kwargs.update(kwargst)
                        
                        Y_true[j, i] = objectives2(kwargs)[0]

                        y_m, var = predict_aug(modeler, gt, kwargs, tid, objtype)
                        # print(kwargs,y_m,var)
                        Y_mean[j, i] = y_m
                        Y_std[j, i]  = np.sqrt(var)
                # print(Y_mean)
                # Sampled points
                samples = np.array(data.P[tid], dtype=float)   # shape (m, 2)
                # print(samples)
                sample_vals = data.O[tid].reshape(-1)  # shape (m,)

                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                titles = ["True Function", "Predicted Mean", "Predicted Std"]

                # --- True function ---
                c0 = axes[0].contourf(X1, X2, Y_true, levels=50, cmap='viridis')
                fig.colorbar(c0, ax=axes[0])
                axes[0].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[0].set_title(titles[0])
                axes[0].set_xlabel("x1")
                axes[0].set_ylabel("x2")

                # --- GP mean ---
                c1 = axes[1].contourf(X1, X2, Y_mean, levels=50, cmap='viridis')
                fig.colorbar(c1, ax=axes[1])
                axes[1].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[1].set_title(titles[1])
                axes[1].set_xlabel("x1")
                axes[1].set_ylabel("x2")

                # --- GP std ---
                c2 = axes[2].contourf(X1, X2, Y_std, levels=50, cmap='magma')
                fig.colorbar(c2, ax=axes[2])
                axes[2].scatter(samples[:,0], samples[:,1], c='r', edgecolor='r', s=1)
                axes[2].set_title(titles[2])
                axes[2].set_xlabel("x1")
                axes[2].set_ylabel("x2")

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.5)
                # input("Press [enter] to continue.")
                if(modelhodlr==True):
                    fig.savefig('obj_2D_%s_N_%s_tol_%s.pdf'%(optimizer,int(NS_input - 1),options['model_hodlrtol']))
                elif(modelsparse==True):    
                    fig.savefig('obj_2D_%s_N_%s_superlu.pdf'%(optimizer,int(NS_input - 1)))
                else:
                    fig.savefig('obj_2D_%s_N_%s.pdf'%(optimizer,int(NS_input - 1)))
                    

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from GPTune.callcgp import cGP
        options['EXAMPLE_NAME_CGP']='GPTune-Demo'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
    
    return stats


def objective_selection():
    objective = input("What Objective Function would you like to use (1, 2, 3 or 4)")
    if ("1" in objective):
        objtype=1
        return objectives1, objtype
    elif ("2" in objective):
        objtype=2
        return objectives2, objtype
    elif ("3" in objective):
        objtype=3
        return objectives3, objtype
    elif ("4" in objective):
        objtype=4
        return objectives4, objtype
    else:
        raise Exception("Invalid objective selection")

    
    
    
import matplotlib.pyplot as plt

def plot_model_results(model_name, objtype, N=None):
    """The plots of plot_training_results.py for one model: its training histories at N, or its optimizer scaling."""
    plot_training_results.MODEL_LABEL = MODELS[model_name][1]
    if N is None:
        plot_optimizer_scaling(objtype, model_dir(model_name))
    else:
        plot_training_histories(objtype, N, model_dir(model_name))


def compare_models(objtype, N, models):
    """
    The test RMSE and CRPS along the trainings of all the models and optimizers at N (the log-likelihoods of
    different models are not comparable) in model_compare_obj<o>_N<N>.pdf/png, and a summary of the runs in
    model_compare_obj<o>_N<N>.csv.
    """
    def first(value):
        return value[0] if isinstance(value, list) else value

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    summary = []
    for model_name, linestyle in zip(models, ('-', '--', ':')):
        for optimizer in plot_training_results.OPTIMIZERS:
            metrics_file = plot_training_results.run_file('training_metrics', objtype, N, optimizer, 'csv', model_dir(model_name))
            stats_file = plot_training_results.run_file('run_stats', objtype, N, optimizer, 'json', model_dir(model_name))
            if not (os.path.exists(metrics_file) and os.path.exists(stats_file)):
                continue
            metrics = np.atleast_1d(np.genfromtxt(metrics_file, delimiter=',', names=True))
            with open(stats_file) as f:
                stats = json.load(f)['stats']
            done = np.isfinite(metrics['rmse'])
            for ax, key in zip(axes, ('rmse', 'crps')):
                ax.plot(metrics['time'][done], metrics[key][done], linestyle=linestyle, marker='o', markersize=3,
                        drawstyle='steps-post' if optimizer in ('mcmc', 'mala') else 'default',
                        color=plot_training_results.COLORS[optimizer], label='%s, %s' % (model_name, plot_training_results.NAMES[optimizer]))
            summary.append([model_name, optimizer, first(stats['time_model']), first(stats['modeling_iteration']),
                            first(stats['time_model_per_likelihoodeval']), first(stats['time_search']),
                            metrics['rmse'][done][-1], metrics['crps'][done][-1]])
    for ax, ylabel in zip(axes, ('test RMSE (%d points)' % NTEST, 'test CRPS (%d points)' % NTEST)):
        ax.set_xlabel('training time (s)')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('Wendland vs INLA sparse GPs, objective %d, N=%d: hyperparameter training' % (objtype, N), fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    prefix = os.path.join(OUTPUT_DIR, 'model_compare_obj%d_N%d' % (objtype, N))
    for extension in ('pdf', 'png'):
        fig.savefig('%s.%s' % (prefix, extension), dpi=150)
    plt.close(fig)
    header = ['model', 'optimizer', 'time_model', 'likelihood_evaluations', 'time_per_evaluation', 'time_search', 'final_test_rmse', 'final_test_crps']
    with open(prefix + '.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for row in summary:
            f.write('%s,%s,%.6f,%d,%.6f,%.6f,%.6e,%.6e\n' % tuple(row))
    print("model comparison, objective %d, N=%d:" % (objtype, N))
    print("  %-9s %-18s %10s %6s %10s %9s %12s %12s" % ('model', 'optimizer', 'model (s)', 'evals', 'per eval', 'search', 'test RMSE', 'test CRPS'))
    for row in summary:
        print("  %-9s %-18s %10.1f %6d %10.3f %9.1f %12.4e %12.4e" % tuple(row))


def plotting(objective, objtype):
    NS = [int(n) + 1 for n in os.environ.get('GP_NS', '102400').split(',')]
    models = os.environ.get('GP_MODELS', 'wendland,inla').split(',')
    optimizers = os.environ.get('GP_OPTIMIZERS', 'gradient,finite difference,mcmc,mala').split(',')
    plotgp = False
    for elem in NS:
        for model_name in models:
            for optimizer in optimizers:
                model_runtime(model="Model_George", obj_func=objective, NS_input=elem, objtype=objtype, modelsparse=True,
                              optimizer=optimizer, plotgp=plotgp, model_name=model_name)
            plot_model_results(model_name, objtype, elem - 1)
        compare_models(objtype, elem - 1, models)
    for model_name in models:
        plot_model_results(model_name, objtype)


def main():
    objective, objtype = objective_selection()
    plotting(objective=objective, objtype=objtype)

    # superlu_terminate()


if __name__ == "__main__":
    main()

