#! /usr/bin/env python3

################################################################################

import numpy as np
import os
import mpi4py
import sys
import time
import argparse


################################################################################

def write_input(params, RUNDIR, niter=1):

    # print("%s/QR.in"%(RUNDIR))
    fin = open("%s/QR.in"%(RUNDIR), 'w')
    fin.write("%d\n"%(len(params) * niter))
    for param in params:
        for k in range(niter):
            # READ( NIN, FMT = 2222 ) FACTOR, MVAL, NVAL, MBVAL, NBVAL, PVAL, QVAL, THRESH
            fin.write("%2s%6d%6d%6d%6d%6d%6d%20.13E\n"%(param[0], param[1], param[2], param[5], param[6], param[9], param[10],param[11]))
    fin.close()

def read_output(params, RUNDIR, niter=1):

    fout = open("%s/QR.out"%(RUNDIR), 'r')
    times = np.ones(len(params))*float('Inf')
    idxparam = 0
    idxiter = 0
    for line in fout.readlines():
        words = line.split()
        # WRITE( NOUT, FMT = 9993 ) 'WALL', M, N, MB, NB, NPROW, NPCOL, WTIME( 1 ), TMFLOPS, PASSED, FRESID
        if (len(words) > 0 and words[0] == "WALL"):
            if (words[9] == "PASSED"):
                m  = int(words[1])
                n  = int(words[2])
                mb = int(words[3])
                nb = int(words[4])
                p  = int(words[5])
                q  = int(words[6])
                thresh = float(words[10])
                mytime = float(words[7])
                while (not ((m == params[idxparam][1])\
                        and (n == params[idxparam][2])\
                        and (mb == params[idxparam][5])\
                        and (nb == params[idxparam][6])\
                        and (p == params[idxparam][9])\
                        and (q == params[idxparam][10]))):
                    idxparam += 1
                if (mytime < times[idxparam]):
                    times[idxparam] = mytime
            idxiter += 1
            if (idxiter >= niter):
                idxparam += 1
                idxiter = 0
    fout.close()

    return times





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str,default='in', help='write input or read output')
    parser.add_argument('-niter', type=int, default=1, help='Number of repeats')
    parser.add_argument('-m', type=int, default=128, help='Number of rows')
    parser.add_argument('-n', type=int, default=128, help='Number of columns')
    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1,help='Number of cores per machine node')
    parser.add_argument('-mb', type=int, default=16, help='Row block size')
    parser.add_argument('-nb', type=int, default=16, help='Column block size')
    parser.add_argument('-nthreads', type=int, default=1, help='OMP Threads')
    parser.add_argument('-nproc', type=int, default=1, help='Number of MPIs')
    parser.add_argument('-p', type=int, default=1, help='Process row count')
    parser.add_argument('-q', type=int, default=1, help='Process column count')
    parser.add_argument('-npernode', type=int, default=1, help='MPI count per node')
    parser.add_argument('-machine', type=str,help='Name of the computer (not hostname)')
    parser.add_argument('-jobid', type=int, default=0, help='ID of the batch job')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()

    niter = args.niter
    mode = args.mode
    m = args.m
    n = args.n
    nodes = args.nodes
    cores = args.cores
    mb = args.mb
    nb = args.nb
    nthreads = args.nthreads
    nproc = args.nproc
    p = args.p
    q = args.q
    npernode = args.npernode

    MACHINE_NAME = args.machine
    JOBID = args.jobid
    TUNER_NAME = 'GPTune'

    os.environ['MACHINE_NAME'] = MACHINE_NAME
    os.environ['TUNER_NAME'] = TUNER_NAME

    ROOTDIR = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
    EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "exp", MACHINE_NAME + '/' + TUNER_NAME))
    if (JOBID==-1):  # -1 is the default value if jobid is not set from command line
        JOBID = os.getpid()
    RUNDIR = os.path.abspath(os.path.join(EXPDIR, str(JOBID)))
    os.makedirs("%s"%(RUNDIR),exist_ok=True)


    params = [('QR', m, n, nodes, cores, mb, nb, nthreads, nproc, p, q, 1., npernode)]

    if(mode=='in'):
        write_input(params, RUNDIR, niter=niter)
    else:
        times = read_output(params, RUNDIR, niter=niter)
        print('PDGEQRF parameter: ',params)
        print('PDGEQRF time: ',times)



