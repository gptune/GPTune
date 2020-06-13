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
################################################################################
"""
Example of invocation of this script:

python hypre.py -nxmax 200 -nymax 200 -nzmax 200 -nodes 1 -cores 32 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -nxmax/nymax/nzmax       maximum number of discretization size for each dimension
    -nodes                   number of compute node
    -cores                   number of cores per node
    -machine                 name of the machine 
    -ntask                   number of different tasks to be tuned
    -nrun                    number of calls per task
    -jobid                   optional, can always be 0
    
Description of the parameters of Hypre AMG:
Task space:
    nx:    problem size in dimension x
    ny:    problem size in dimension y
    nz:    problem size in dimension z
    cx:    diffusion coefficient for d^2/dx^2
    cy:    diffusion coefficient for d^2/dy^2
    cz:    diffusion coefficient for d^2/dz^2
    ax:    convection coefficient for d/dx
    ay:    convection coefficient for d/dy
    az:    convection coefficient for d/dz
Input space:
    Px:                processor topology, with NProc = Px*Py*Pz
    Py:                processor topology, with NProc = Px*Py*Pz
    Pz:                processor topology, with NProc = Px*Py*Pz
    strong_threshold:  AMG strength threshold
    trunc_factor:      Truncation factor for interpolation
    P_max_elmts:       Max number of elements per row for AMG interpolation
    coarsen_type:      Defines which parallel coarsening algorithm is used
    relax_type:        Defines which smoother to be used
    smooth_type:       Enables the use of more complex smoothers
    smooth_num_levels: Number of levels for more complex smoothers
    interp_type:       Defines which parallel interpolation operator is used  
    agg_num_levels:    Number of levels of aggressive coarsening
"""
from hypredriver import hypredriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import sys, os, re
import numpy as np
import time
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter

# import mpi4py
# from mpi4py import MPI

# add GPTunde path in front of all python pkg path
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../hypre_driver/"))

solver = 3 # Bommer AMG
max_setup_time = 100.
max_solve_time = 100.
coeffs_c = "-c 1 1 1 " # specify c-coefficients in format "-c 1 1 1 " 
coeffs_a = "-a 0 0 0 " # specify a-coefficients in format "-a 1 1 1 " leave as empty string for laplacian and Poisson problems
problem_name = "-laplacian " # "-difconv " for convection-diffusion problems to include the a coefficients

# define objective function
def objectives(point):
    # task params 
    nx = point['nx']
    ny = point['ny']
    nz = point['nz']
    # tuning params / input params
    Px = point['Px']
    Py = point['Py']
    Pz = point['Pz']
    strong_threshold = point['strong_threshold']
    trunc_factor = point['trunc_factor']
    P_max_elmts = point['P_max_elmts']
    coarsen_type = point['coarsen_type']
    relax_type = point['relax_type']
    smooth_type = point['smooth_type']
    smooth_num_levels = point['smooth_num_levels']
    interp_type = point['interp_type']
    agg_num_levels = point['agg_num_levels']

    NProc = Px*Py*Pz
    # CoarsTypes = {0:"-cljp ", 1:"-ruge ", 2:"-ruge2b ", 3:"-ruge2b ", 4:"-ruge3c ", 6:"-falgout ", 8:"-pmis ", 10:"-hmis "}
    # CoarsType = CoarsTypes[coarsen_type]

    # call Hypre 
    params = [(nx, ny, nz, coeffs_a, coeffs_c, problem_name, solver,
               Px, Py, Pz, strong_threshold, 
               trunc_factor, P_max_elmts, coarsen_type, relax_type, 
               smooth_type, smooth_num_levels, interp_type, agg_num_levels)]
    runtime = hypredriver(params, niter=1, JOBID=JOBID)
    print(params, ' hypre time: ', runtime)

    return runtime


def models(): # todo
    pass

def main(): 
    global nodes
    global cores
    global JOBID
    global nprocmax
    global nprocmin

    # Parse command line arguments
    args = parse_args()

    nxmax = args.nxmax
    nymax = args.nymax
    nzmax = args.nzmax
    nodes = args.nodes
    cores = args.cores
    machine = args.machine
    ntask = args.ntask
    nruns = args.nruns
    JOBID = args.jobid
    TUNER_NAME = args.optimization
    TLA = False

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores-1 
    nprocmin = nodes
    
    nxmin = 20
    nymin = 20
    nzmin = 20
    nx = Integer(nxmin, nxmax, transform="normalize", name="nx")
    ny = Integer(nymin, nymax, transform="normalize", name="ny")
    nz = Integer(nzmin, nzmax, transform="normalize", name="nz")
    Px = Integer(1, nprocmax, transform="normalize", name="Px")
    Py = Integer(1, nprocmax, transform="normalize", name="Py")
    Pz = Integer(1, nprocmax, transform="normalize", name="Pz")
    strong_threshold = Real(0, 1, transform="normalize", name="strong_threshold")
    trunc_factor =  Real(0, 1, transform="normalize", name="trunc_factor")
    P_max_elmts = Integer(1, 12,  transform="normalize", name="P_max_elmts")
    coarsen_type = Categoricalnorm (['0', '1', '2', '3', '4', '6', '8', '10'], transform="onehot", name="coarsen_type")
    relax_type = Categoricalnorm (['-1', '0', '6', '8', '16', '18'], transform="onehot", name="relax_type")
    smooth_type = Categoricalnorm (['5', '6', '7', '8', '9'], transform="onehot", name="smooth_type")
    smooth_num_levels = Integer(0, 5,  transform="normalize", name="smooth_num_levels")
    interp_type = Categoricalnorm (['0', '3', '4', '5', '6', '8', '12'], transform="onehot", name="interp_type")
    agg_num_levels = Integer(0, 5,  transform="normalize", name="agg_num_levels")
    r = Real(float("-Inf"), float("Inf"), name="r")
    
    IS = Space([nx, ny, nz])
    PS = Space([Px, Py, Pz, strong_threshold, trunc_factor, P_max_elmts, coarsen_type, relax_type, smooth_type, smooth_num_levels, interp_type, agg_num_levels])
    OS = Space([r])
    
    # Question: how to set constraints
    cst1 = f"Px * Py * Pz <= {nprocmax}"
    cst2 = f"Px * Py * Pz >= {nprocmin}"
    constraints = {"cst1": cst1, "cst2": cst2}
    
    # cst1 = f"1 <= Px <= {nprocmax}"
    # cst2 = f"1 <= Py <= {nprocmax}"
    # cst3 = f"1 <= Pz <= {nprocmax}"
    # constraints = {"cst1": cst1, "cst2": cst2, "cst3": cst3}
    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None) # no performance model
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class '] = 'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)
    
    
    """ Intialize the tuner with existing data stored as last check point"""
    try:
        data = pickle.load(open('Data_nodes_%d_cores_%d_nxmax_%d_nymax_%d_nzmax_%d_machine_%s_jobid_%d.pkl' % (nodes, cores, nxmax, nymax, nzmax, machine, JOBID), 'rb'))
        giventask = data.I
    except (OSError, IOError) as e:
        data = Data(problem)
        giventask = [[randint(nxmin,nxmax),randint(nymin,nymax),randint(nzmin,nzmax)] for i in range(ntask)]

    # giventask = [[50, 60, 80], [60, 80, 100]]
    # # the following will use only task lists stored in the pickle file
    # data = Data(problem)


    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nruns
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)
        
        """ Dump the data to file as a new check point """
        pickle.dump(data, open('Data_nodes_%d_cores_%d_nxmax_%d_nymax_%d_nzmax_%d_machine_%s_jobid_%d.pkl' % (nodes, cores, nxmax, nymax, nzmax, machine, JOBID), 'wb'))
        
        """ Dump the tuner to file for TLA use """
        pickle.dump(gt, open('MLA_nodes_%d_cores_%d_nxmax_%d_nymax_%d_nzmax_%d_machine_%s_jobid_%d.pkl' % (nodes, cores, nxmax, nymax, nzmax, machine, JOBID), 'wb'))


        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if TLA is True:
            """ Call TLA for 2 new tasks using the constructed LCM model"""
            newtask = [[50, 50, 60], [80, 60, 70]]
            (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])

    
    if(TUNER_NAME=='opentuner'):
        NI = ntask
        NS = nruns
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = ntask
        NS = nruns
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    nx:%d ny:%d nz:%d" % (data.I[tid][0], data.I[tid][1], data.I[tid][2]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


def parse_args():
    parser = argparse.ArgumentParser()
    # Problem related arguments
    parser.add_argument('-nxmax', type=int, default=50, help='discretization size in dimension x')
    parser.add_argument('-nymax', type=int, default=50, help='discretization size in dimension y')
    parser.add_argument('-nzmax', type=int, default=50, help='discretization size in dimension y')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    # Algorithm related arguments
    # parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-optimization', type=str,help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, default=-1, help='Number of runs per task')
    # parser.add_argument('-truns', type=int, default=-1, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
