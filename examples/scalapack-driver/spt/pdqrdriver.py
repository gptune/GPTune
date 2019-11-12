#! /usr/bin/env python3

################################################################################

import numpy as np
import os
import mpi4py
from mpi4py import MPI
import sys
################################################################################

# Paths

MACHINE_NAME = 'tmp'
TUNER_NAME = 'tmp'
ROOTDIR = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
# SPTDIR = os.path.abspath(os.path.join(ROOTDIR, "spt"))
# SRCDIR = os.path.abspath(os.path.join(ROOTDIR, "src"))
BINDIR = os.path.abspath(os.path.join(ROOTDIR, "bin", MACHINE_NAME))
EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "exp", MACHINE_NAME + '/' + TUNER_NAME))

################################################################################

def compile():

    global ROOTDIR

    os.system("cd %s; make;"%(ROOTDIR))

def clean():

    global ROOTDIR

    os.system("cd %s; make clean;"%(ROOTDIR))

################################################################################

def write_input(params, RUNDIR, niter=1):

    fin = open("%s/QR.in"%(RUNDIR), 'w')
    fin.write("%d\n"%(len(params) * niter))
    for param in params:
        for k in range(niter):
            # READ( NIN, FMT = 2222 ) FACTOR, MVAL, NVAL, MBVAL, NBVAL, PVAL, QVAL, THRESH
            fin.write("%2s%6d%6d%6d%6d%6d%6d%18.13F\n"%(param[0], param[1], param[2], param[5], param[6], param[9], param[10], param[11]))
    fin.close()

def execute(nproc, nth, RUNDIR):

    #XXX To be removed
    def v_sequential():
        return os.system("cd %s; export OMP_PLACES=threads; export OMP_PROC_BIND=spread; export OMP_NUM_THREADS=1; %s/pdqrdriver 2> err;"%(RUNDIR, BINDIR))

    def v_parallel():
        # os.system("cd %s;"%(RUNDIR)) 
        # print('nimdda',RUNDIR)
        comm = MPI.COMM_SELF.Spawn("%s/pdqrdriver"%(BINDIR), args="%s/"%(RUNDIR), maxprocs=nproc)
        comm.Disconnect()
        return 0


        # return os.system("cd %s; export OMP_PLACES=threads; export OMP_PROC_BIND=spread; export OMP_NUM_THREADS=%d; mpirun -c %d -n %d %s/pdqrdriver 2>> QR.err &  wait;"%(RUNDIR, nth, 2*nth, nproc, BINDIR))

##    err = v_sequential()
    err = v_parallel()
    if (err != 0):
        print("Error running ScaLAPACK: ", err)

    return

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

def pdqrdriver(params, niter=10):

    global EXPDIR 
    global BINDIR
    global ROOTDIR

    MACHINE_NAME = os.environ['MACHINE_NAME']
    TUNER_NAME = os.environ['TUNER_NAME']
    BINDIR = os.path.abspath(os.path.join(ROOTDIR, "bin", MACHINE_NAME))
    EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "exp", MACHINE_NAME + '/' + TUNER_NAME))


    RUNDIR = os.path.abspath(os.path.join(EXPDIR, str(os.getpid())))
    os.system("mkdir -p %s"%(RUNDIR))
    # print('nima',RUNDIR)

    dtype = [("fac", 'U10'), ("m", int), ("n", int), ("nodes", int), ("cores", int), ("mb", int), ("nb", int), ("nth", int), ("nproc", int), ("p", int), ("q", int), ("thresh", float)]
    params = np.array(params, dtype=dtype)
    perm = np.argsort(params, order=["nproc", "nth"])
    invperm = np.argsort(perm)
    idxproc = 8
    idxth = 7
    times = np.array([])
    k_beg = 0
    k_end = 1
    while (k_end < len(params)):
        if ((params[perm[k_beg]][idxproc] != params[perm[k_end]][idxproc]) or (params[perm[k_beg]][idxth] != params[perm[k_end]][idxth])):
            write_input(params[perm[k_beg:k_end]], RUNDIR, niter=niter)
            execute(params[perm[k_beg]][idxproc], params[perm[k_beg]][idxth], RUNDIR)
            times2 = read_output(params[perm[k_beg:k_end]], RUNDIR, niter=niter)
            times = np.concatenate((times, times2))
            k_beg = k_end
        k_end += 1
    if (k_beg < len(params)):
        write_input(params[perm[k_beg:k_end]], RUNDIR, niter=niter)
        execute(params[perm[k_beg]][idxproc], params[perm[k_beg]][idxth], RUNDIR)
        times2 = read_output(params[perm[k_beg:k_end]], RUNDIR, niter=niter)
        times = np.concatenate((times, times2))
    times = times[invperm]

    os.system('rm -fr %s'%(RUNDIR))

    return times

if __name__ == "__main__":

    # Test

#    compile()
    params = [('QR', 1000, 1000, 1, 32, 32, 32, 2, 2, 2, 1, 1.),\
              ('QR', 1000, 1000, 1, 32, 32, 32, 1, 1, 1, 1, 1.),\
              ('QR', 1000, 1000, 1, 32, 32, 32, 2, 1, 1, 1, 1.),\
              ('QR',  100,  100, 1, 32, 32, 32, 2, 1, 1, 1, 1.),\
              ('QR', 1000, 1000, 1, 32, 32, 32, 1, 2, 2, 1, 1.),\
              ('QR',  100,  100, 1, 32, 32, 32, 1, 2, 2, 1, 1.)]
    times = pdqrdriver(params, niter=3)
    print(times)
#    clean()

# vim: set ft=python:
