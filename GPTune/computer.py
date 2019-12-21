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

import numpy as np
from problem import Problem
from data import Data
from typing import Collection, Callable
import mpi4py
from mpi4py import MPI
import os
import sys


class Computer(object):

    def __init__(self, nodes : int = 1, cores : int = 1, hosts : Collection = None):

        self.nodes = nodes
        self.cores = cores
        self.hosts = hosts
        if (hosts != None and nodes != len(hosts)):
            raise Exception('The number of elements in "hosts" does not match with the number of "nodes"')

    def evaluate_constraints(self, problem : Problem, point : Collection, inputs_only : bool = False, **kwargs):  # point is in the original spaces

#       kwargs['constraints_evaluation_parallelism']

        # points can be either a dict or a list of dicts on which to iterate

        cond = True
        for (cstname, cst) in problem.constraints.items():
            if (isinstance(cst, str)):
                try:
                    # {} has to be the global argument to eval
                    # and point the local one, otherwise,
                    # point will be corrupted / updated by eval
                    cond = eval(cst, {}, point)
                except Exception as inst:
                    if (inputs_only and isinstance(inst, NameError)):
                        pass
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            else:
                try:
                    kwargs2 = {}
                    sig = inspect.signature(cst)
                    for varname in point:
                        if (varname in sig.parameters):
                            kwargs2[varname] = point[varname]
                    cond = cst(**kwargs2)
                except Exception as inst:
                    if (isinstance(inst, TypeError)):
                        lst = inst.__str__().split()
                        if (len(lst) >= 5 and lst[1] == 'missing' and lst[3] == 'required' and lst[4] == 'positional'):
                            pass
                        else:
                            raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            if (not cond):
                break

        return cond


    def evaluate_objective(self, problem : Problem, I : np.ndarray = None, P : Collection[np.ndarray] = None, **kwargs):  # P and I are in the normalized space

#        kwargs['objective_evaluation_parallelism'])

        O = []
        for i in range(len(I)):
            t = I[i]
            I_orig = problem.IS.inverse_transform(np.array(t, ndmin=2))[0]		
            kwargst = {problem.IS[k].name: I_orig[k] for k in range(problem.DI)}
            P2 = P[i]
            O2 = []
            for j in range(len(P2)):
                x = P2[j]
                x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]		
                kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                # print(kwargs)
                kwargs.update(kwargst)
                # print(kwargs)
                o = problem.objective(kwargs)
                O2.append(o)
            O.append(np.array(O2).reshape((len(O2), problem.DO)))

        return O

    def spawn(self, executable, nproc, nth, args=None, kwargs=None): 

        # XXX
#        check_mpi()
#        mpi_info = MPI.Info.Create()
#        mpi_info.Set("add-hostfile", "slurm.hosts")
#        mpi_info.Set("host", "slurm.hosts")
         

        print('exec', executable, 'args', args, 'nproc', nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
#        comm = MPI.COMM_SELF.Spawn(executable, args=args, maxprocs=nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
#        comm = MPI.COMM_SELF.Spawn('/usr/common/software/python/3.7-anaconda-2019.07/bin/python', args=executable, maxprocs=nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
        
        info = MPI.Info.Create()
        info.Set('env', 'OMP_NUM_THREADS=%d\n' %(nth))        
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=executable, maxprocs=nproc,info=info)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
        # process_rank = comm.Get_rank()
        # process_count = comm.Get_size()
        # process_host = MPI.Get_processor_name()
        # print('manager',process_rank, process_count, process_host)
        return comm

#print(MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name())
