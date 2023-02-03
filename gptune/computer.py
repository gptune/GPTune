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
from .problem import Problem
from .data import Data
from .database import HistoryDB
from typing import Collection, Callable
import os
import sys
import concurrent
from concurrent import futures

from pathlib import Path
import importlib
import inspect

class Computer(object):

    def __init__(self, nodes : int = 1, cores : int = 1, hosts : Collection = None):

        self.nodes = nodes
        self.cores = cores
        self.hosts = hosts
        if (hosts != None and nodes != len(hosts)):
            raise Exception('The number of elements in "hosts" does not match with the number of "nodes"')

    def evaluate_constraints(self, problem, point : Collection, inputs_only : bool = False, **kwargs):  # point is in the original spaces

#       kwargs['constraints_evaluation_parallelism']

        # points can be either a dict or a list of dicts on which to iterate
        if(problem.constants is not None):
            point.update(problem.constants)
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
                    if(hasattr(problem, 'driverabspath')): # differentiate between Problem and TuningProblem 
                        if(problem.driverabspath is not None):
                            modulename = Path(problem.driverabspath).stem  # get the driver name excluding all directories and extensions
                            sys.path.append(problem.driverabspath) # add path to sys
                            module = importlib.import_module(modulename) # import driver name as a module
                            cst = getattr(module, cstname)
                        else:
                            raise Exception('the driverabspath is required for the constraints')        
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


    def evaluate_objective(self, problem : Problem, I : np.ndarray = None, P : Collection[np.ndarray] = None, D: Collection[dict] = None, history_db : HistoryDB = None, options: dict=None, is_pilot = False): # P and I are in the normalized space

        O = []

        for i in range(len(I)):
            T2 = I[i]
            P2 = P[i]
            if D is not None:
                D2 = D[i]
            else:
                D2 = None
            if(options['RCI_mode']==False):
                O2 = self.evaluate_objective_onetask(problem=problem, i_am_manager=True, T2=T2, P2=P2, D2=D2, history_db=history_db, options=options, is_pilot=is_pilot)

                tmp = np.array(O2).reshape((len(O2), problem.DO))
                O.append(tmp.astype(np.double))   #YL: convert single, double or int to double types

            else:
                tmp = np.empty( shape=(len(P2), problem.DO))
                tmp[:] = np.NaN
                O.append(tmp.astype(np.double))   #YL: NaN indicates that the evaluation data is needed by GPTune

                if history_db is not None:

                    if is_pilot == True:
                        modeling = "Pilot"
                    else:
                        if options["TLA_method"] == None:
                            if len(I) == 1:
                                modeling = "SLA_GP"
                            elif len(I) > 1:
                                modeling = "MLA_LCM"
                        elif options["TLA_method"] == "Regression":
                            modeling = "TLA_RegressionSum"
                        elif options["TLA_method"] == "Sum":
                            modeling = "TLA_Sum"
                        elif options["TLA_method"] == "Stacking":
                            modeling = "TLA_Stacking"
                        elif options["TLA_method"] == "LCM_BF":
                            modeling = "TLA_LCM_BF"
                        elif options["TLA_method"] == "LCM":
                            modeling = "TLA_LCM"
                        else:
                            if len(I) == 1:
                                modeling = "SLA_GP"
                            elif len(I) > 1:
                                modeling = "MLA_LCM"

                    history_db.store_func_eval(problem = problem,\
                            task_parameter = I[i], \
                            tuning_parameter = P[i],\
                            evaluation_result = tmp,\
                            evaluation_detail = tmp,\
                            source = "RCI_measure",\
                            modeling = modeling,\
                            model_class = options["model_class"])

        if(options['RCI_mode']==True):
            print('RCI: GPTune returns\n')
            exit()

        return O

    def evaluate_objective_TLA(self, problem : Problem, I : np.ndarray = None, P : Collection[np.ndarray] = None, D: Collection[dict] = None, history_db : HistoryDB = None, options: dict=None, models_transfer : list = None):  # P and I are in the normalized space

        num_given_tasks = len(I)
        num_source_tasks = len(models_transfer)
        num_target_tasks = num_given_tasks - num_source_tasks

        O = []
        for i in range(len(I)):
            T2 = I[i]
            P2 = P[i]
            if D is not None:
                D2 = D[i]
            else:
                D2 = None
            if(options['RCI_mode']==False):
                if i >= num_target_tasks:
                    O2 = self.model_predict_objective_onetask(problem=problem, i_am_manager=True, T2=T2, P2=P2, D2=D2, history_db=history_db, options=options, model_transfer=models_transfer[i-num_target_tasks], source = "model")
                else:
                    O2 = self.evaluate_objective_onetask(problem=problem, i_am_manager=True, T2=T2, P2=P2, D2=D2, history_db=history_db, options=options)

                tmp = np.array(O2).reshape((len(O2), problem.DO))
                O.append(tmp.astype(np.double))   #YL: convert single, double or int to double types

            else:
                if i >= num_target_tasks:
                    O2 = self.model_predict_objective_onetask(problem=problem, i_am_manager=True, T2=T2, P2=P2, D2=D2, history_db=history_db, options=options, model_transfer=models_transfer[i-num_target_tasks], source = "RCI_model")
                else:
                    tmp = np.empty( shape=(len(P2), problem.DO))
                    tmp[:] = np.NaN
                    O.append(tmp.astype(np.double))   #YL: NaN indicates that the evaluation data is needed by GPTune

                    if history_db is not None:

                        if options["TLA_method"] == None:
                            if len(I) == 1:
                                modeling = "SLA_GP"
                            elif len(I) > 1:
                                modeling = "MLA_LCM"
                        elif options["TLA_method"] == "Regression":
                            modeling = "TLA_RegressionSum"
                        elif options["TLA_method"] == "Sum":
                            modeling = "TLA_Sum"
                        elif options["TLA_method"] == "Stacking":
                            modeling = "TLA_Stacking"
                        elif options["TLA_method"] == "LCM_BF":
                            modeling = "TLA_LCM_BF"
                        elif options["TLA_method"] == "LCM":
                            modeling = "TLA_LCM"
                        else:
                            if len(I) == 1:
                                modeling = "SLA_GP"
                            elif len(I) > 1:
                                modeling = "MLA_LCM"

                        history_db.store_func_eval(problem = problem,\
                                task_parameter = I[i], \
                                tuning_parameter = P[i],\
                                evaluation_result = tmp,\
                                evaluation_detail = tmp,\
                                source = "RCI_measure",\
                                modeling = modeling,\
                                model_class = options["model_class"])

        if(options['RCI_mode']==True):
            print('RCI: GPTune returns\n')
            exit()

        return O

    def evaluate_objective_onetask(self, problem : Problem, pids : Collection[int] = None, i_am_manager : bool = True, T2 : np.ndarray=None, P2 : np.ndarray=None, D2 : dict=None, history_db : HistoryDB=None, options:dict=None, is_pilot=False):  # T2 and P2 are in the normalized space

        I_orig = problem.IS.inverse_transform(np.array(T2, ndmin=2))[0]

        if(problem.driverabspath is not None and options['distributed_memory_parallelism']):
            modulename = Path(problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module
            # func = getattr(module, funcName)
        else:
            module =problem

        O2=[]
        kwargst = {problem.IS[k].name: I_orig[k] for k in range(problem.DI)}

        if (pids is None):
            pids = list(range(len(P2)))

        if (options['distributed_memory_parallelism'] and options['objective_evaluation_parallelism'] and i_am_manager):
            import mpi4py
            from mpi4py import MPI
            if(problem.driverabspath is None):
                raise Exception('objective_evaluation_parallelism and distributed_memory_parallelism require passing driverabspath to GPTune')

            nproc = min(options['objective_multisample_processes'],len(P2))
            mpi_comm = self.spawn(__file__, nproc, nthreads=1, kwargs=options)
            kwargs_tmp = options
            if "mpi_comm" in kwargs_tmp:
                del kwargs_tmp["mpi_comm"]   # mpi_comm is not picklable
            _ = mpi_comm.bcast((self, problem,P2, D2, I_orig, pids, history_db, kwargs_tmp), root=mpi4py.MPI.ROOT)

            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()

            # reordering is needed as tmpdata[p] stores p, p+nproc, p+2nproc, ...
            Otmp=[]
            offset=[0] * (nproc+1)
            for p in range(int(nproc)):
                Otmp = Otmp + tmpdata[p]
                offset[p+1]=offset[p]+len(tmpdata[p])
            for it in range(len(tmpdata[0])):
                for p in range(int(nproc)):
                    if(len(O2)<len(P2)):
                        O2.append(Otmp[offset[p]+it])

        elif (options['shared_memory_parallelism'] and options['objective_evaluation_parallelism']):
            with concurrent.futures.ThreadPoolExecutor(max_workers = options['objective_multisample_threads']) as executor:
                def fun(pid):
                    x = P2[pid]
                    x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                    kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                    kwargs.update(kwargst)
                    kwargs.update(D2)
                    if(problem.constants is not None):
                        kwargs.update(problem.constants)
                    # print(kwargs)
                    o_ = module.objectives(kwargs)

                    # return statement format 1: "return [...]" (return from an objective function)
                    # return statement format 2: "return {...}" (return from a surrogate model black-box function)
                    if type(o_) == dict or type(o_) == list:
                        o = o_
                        additional_output = None
                    # output format 3: "return [...], {...}"
                    # (return from an objective function if the user wants to pass some additional information using a dictionary)
                    elif type(o_) == tuple:
                        o = o_[0]
                        additional_output = o_[1]

                    o_eval = []

                    if type(o) == type({}): # predicted by surrogate model black-box function
                        source = o["source"]
                        o_eval = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                        o_detail = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                    elif type(o) == list: # measured from the objective function
                        source = "measure"
                        for i in range(len(o)):
                            if type(o[i]) == type([]):
                                o_eval.append(np.average(o[i]))
                            else:
                                o_eval.append(o[i])
                        o_detail = o

                    if history_db is not None:
                        if is_pilot == True:
                            modeling = "Pilot"
                        else:
                            if options["TLA_method"] == None:
                                if problem.DI == 1:
                                    modeling = "SLA_GP"
                                elif problem.DI > 1:
                                    modeling = "MLA_LCM"
                            elif options["TLA_method"] == "Regression":
                                modeling = "TLA_RegressionSum"
                            elif options["TLA_method"] == "Sum":
                                modeling = "TLA_Sum"
                            elif options["TLA_method"] == "Stacking":
                                modeling = "TLA_Stacking"
                            elif options["TLA_method"] == "LCM_BF":
                                modeling = "TLA_LCM_BF"
                            elif options["TLA_method"] == "LCM":
                                modeling = "TLA_LCM"
                            else:
                                if problem.DI == 1:
                                    modeling = "SLA_GP"
                                elif problem.DI > 1:
                                    modeling = "MLA_LCM"

                        history_db.store_func_eval(problem = problem,\
                                task_parameter = T2, \
                                tuning_parameter = [P2[pid]],\
                                evaluation_result = [o_eval], \
                                evaluation_detail = [o_detail], \
                                additional_output = additional_output,
                                source = source,\
                                modeling = modeling,\
                                model_class = options["model_class"])

                    return o

                O2 = list(executor.map(fun, pids, timeout=None, chunksize=1))

        else:

            for j in pids:
                x = P2[j]
                x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                kwargs.update(kwargst)
                if(problem.constants is not None):
                    kwargs.update(problem.constants)
                if D2 is not None:
                    kwargs.update(D2)
                o_ = module.objectives(kwargs)

                # return statement format 1: "return [...]" (return from an objective function)
                # return statement format 2: "return {...}" (return from a surrogate model black-box function)
                if type(o_) == dict or type(o_) == list or type(o_) == np.ndarray:
                    o = o_
                    additional_output = None
                # output format 3: "return [...], {...}"
                # (return from an objective function if the user wants to pass some additional information using a dictionary)
                elif type(o_) == tuple:
                    o = o_[0]
                    additional_output = o_[1]

                o_eval = []

                if type(o) == dict: # predicted by model
                    source = o["source"]
                    o_eval = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                    o_detail = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                elif type(o) == list or type(o) == np.ndarray: # measured from the objective function
                    source = "measure"
                    for i in range(len(o)):
                        if type(o[i]) == type([]):
                            o_eval.append(np.average(o[i]))
                        else:
                            o_eval.append(o[i])
                    o_detail = o

                if history_db is not None:
                    if is_pilot == True:
                        modeling = "Pilot"
                    else:
                        if options["TLA_method"] == None:
                            if problem.DI == 1:
                                modeling = "SLA_GP"
                            elif problem.DI > 1:
                                modeling = "MLA_LCM"
                        elif options["TLA_method"] == "Regression":
                            modeling = "TLA_RegressionSum"
                        elif options["TLA_method"] == "Sum":
                            modeling = "TLA_Sum"
                        elif options["TLA_method"] == "Stacking":
                            modeling = "TLA_Stacking"
                        elif options["TLA_method"] == "LCM_BF":
                            modeling = "TLA_LCM_BF"
                        elif options["TLA_method"] == "LCM":
                            modeling = "TLA_LCM"
                        else:
                            if problem.DI == 1:
                                modeling = "SLA_GP"
                            elif problem.DI > 1:
                                modeling = "MLA_LCM"

                    history_db.store_func_eval(problem = problem,\
                            task_parameter = T2, \
                            tuning_parameter = [P2[j]],\
                            evaluation_result = [o_eval], \
                            evaluation_detail = [o_detail], \
                            additional_output = additional_output,
                            source = source,\
                            modeling = modeling,\
                            model_class = options["model_class"])

                O2.append(o_eval)

        return O2

    def model_predict_objective_onetask(self, problem : Problem, pids : Collection[int] = None, i_am_manager : bool = True, T2 : np.ndarray=None, P2 : np.ndarray=None, D2 : dict=None, history_db : HistoryDB=None, options:dict=None, model_transfer:list=None, source:str="model"):  # T2 and P2 are in the normalized space

        I_orig = problem.IS.inverse_transform(np.array(T2, ndmin=2))[0]

        if(problem.driverabspath is not None and options['distributed_memory_parallelism']):
            modulename = Path(problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module
            # func = getattr(module, funcName)
        else:
            module =problem

        O2=[]
        kwargst = {problem.IS[k].name: I_orig[k] for k in range(problem.DI)}

        if (pids is None):
            pids = list(range(len(P2)))

        if (options['distributed_memory_parallelism'] and options['objective_evaluation_parallelism'] and i_am_manager):
            print ("unsupported yet")
        elif (options['shared_memory_parallelism'] and options['objective_evaluation_parallelism']):
            print ("unsupported yet")
        else:
            for j in pids:
                x = P2[j]
                x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]
                kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                kwargs.update(kwargst)
                if(problem.constants is not None):
                    kwargs.update(problem.constants)
                if D2 is not None:
                    kwargs.update(D2)
                o_ = model_transfer(kwargs)
                print (o_)
                if type(o_) == type({}) or len(o_) == 1:
                    o = o_
                    additional_output = None
                elif len(o_) == 2:
                    o = o_[0]
                    additional_output = o_[1]

                o_eval = []

                o_eval = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                o_detail = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]

                #if type(o) == type({}): # predicted by model
                #    source = o["source"]
                #    o_eval = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                #    o_detail = [o[problem.OS[k].name][0][0] for k in range(len(problem.OS))]
                #else: # type(o) == type([]): # list
                #    source = "measure"
                #    for i in range(len(o)):
                #        if type(o[i]) == type([]):
                #            o_eval.append(np.average(o[i]))
                #        else:
                #            o_eval.append(o[i])
                #    o_detail = o

                if source == "RCI_model":
                    if history_db is not None:

                        if options["TLA_method"] == None:
                            if problem.DI == 1:
                                modeling = "SLA_GP"
                            elif problem.DI > 1:
                                modeling = "MLA_LCM"
                        elif options["TLA_method"] == "Regression":
                            modeling = "TLA_RegressionSum"
                        elif options["TLA_method"] == "Sum":
                            modeling = "TLA_Sum"
                        elif options["TLA_method"] == "Stacking":
                            modeling = "TLA_Stacking"
                        elif options["TLA_method"] == "LCM_BF":
                            modeling = "TLA_LCM_BF"
                        elif options["TLA_method"] == "LCM":
                            modeling = "TLA_LCM"
                        else:
                            if problem.DI == 1:
                                modeling = "SLA_GP"
                            elif problem.DI > 1:
                                modeling = "MLA_LCM"

                        history_db.store_func_eval(problem = problem,\
                                task_parameter = T2, \
                                tuning_parameter = [P2[j]],\
                                evaluation_result = [o_eval], \
                                evaluation_detail = [o_detail], \
                                additional_output = additional_output,
                                source = source,\
                                modeling = modeling,\
                                model_class = options["model_class"])
                                #np.array(O2_).reshape((len(O2_), problem.DO)), \

                O2.append(o_eval)

        return O2


    def spawn(self, executable, nproc, nthreads, npernode=None, args=None, kwargs=None):
        import mpi4py
        from mpi4py import MPI
        print('exec', executable, 'args', args, 'nproc', nproc)

        npernodes=npernode
        if(npernode is None):
            npernodes=self.cores

        info = mpi4py.MPI.Info.Create()
#        info.Set("add-hostfile", "slurm.hosts")
#        info.Set("host", "slurm.hosts")
        info.Set('env', 'OMP_NUM_THREADS=%d\n' %(nthreads))
        info.Set('npernode','%d'%(npernodes))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works


        comm = mpi4py.MPI.COMM_SELF.Spawn(sys.executable, args=executable, maxprocs=nproc,info=info)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
        # process_rank = comm.Get_rank()
        # process_count = comm.Get_size()
        # process_host = mpi4py.MPI.Get_processor_name()
        # print('manager',process_rank, process_count, process_host)
        return comm


if __name__ == '__main__':
    import mpi4py
    from mpi4py import MPI

    def objectives(point):
        print('this is a dummy definition')
        return point
    def models(point):
        print('this is a dummy definition')
        return point

    mpi_comm = mpi4py.MPI.Comm.Get_parent()
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    (computer, problem,P2, D2, I_orig, pids, history_db, kwargs) = mpi_comm.bcast(None, root=0)
    pids_loc = pids[mpi_rank:len(pids):mpi_size]
    T2 = problem.IS.transform([I_orig])[0]

    tmpdata = computer.evaluate_objective_onetask(problem, pids_loc, False, T2, P2, D2, history_db, options=kwargs)
    res = mpi_comm.gather(tmpdata, root=0)
    mpi_comm.Disconnect()
