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
from autotune.problem import TuningProblem
from .problem import Problem
from .options import Options
from .computer import Computer
from .database import HistoryDB
from typing import Collection

import skopt.space
from skopt.space import *
from data import *
import math
import opentuner
import argparse
import functools
import time

####################################################################################################

class OpenTunerInterface(opentuner.MeasurementInterface):

    def __init__(self, args):

        super(OpenTunerInterface, self).__init__(args)
        self.args = args
        self.X = []
        self.Y = []
        self.timefun = 0

    def manipulator(self):

        """
        Define the search space by creating a ConfigurationManipulator
        """

        manipulator = opentuner.ConfigurationManipulator()
        for n,p in zip(self.args.tp.parameter_space.dimension_names,self.args.tp.parameter_space.dimensions):
            if (isinstance(p, Real)):
                manipulator.add_parameter(opentuner.FloatParameter(n, p.bounds[0], p.bounds[1]))
            elif (isinstance(p, Integer)):
                manipulator.add_parameter(opentuner.IntegerParameter(n, p.bounds[0], p.bounds[1]))
            elif (isinstance(p, Categorical)):
                manipulator.add_parameter(opentuner.search.manipulator.EnumParameter(n, list(p.bounds)))
            else:
                raise Exception("Unknown parameter type")

        self.cpt = 0

        return manipulator

    def run(self, desired_result, input, limit):

        """
        Run a given configuration then return performance
        """
        t1 = time.time_ns()
        # Extract parameters

        cfg = desired_result.configuration.data
        t = self.args.t
        x = [cfg[p] for p in self.args.tp.parameter_space.dimension_names]
        kwargs = {d.name: x[i] for (i, d) in enumerate(self.args.tp.parameter_space)}
        kwargs2 = {d.name: t[i] for (i, d) in enumerate(self.args.tp.input_space)}
        kwargs2.update(kwargs)
        check_constraints = functools.partial(self.args.computer.evaluate_constraints, self.args.tp, inputs_only = False, kwargs = kwargs)
        cond = check_constraints(kwargs2)
        if (cond):
            #y = float(self.args.tp.objective(kwargs2)[0])

            transform_T = self.args.tp.input_space.transform([t])[0]
            transform_X = self.args.tp.parameter_space.transform([x])
            result = self.args.computer.evaluate_objective_onetask(
                    problem = self.args.problem,
                    i_am_manager = True,
                    T2 = transform_T,
                    P2 = transform_X,
                    D2 = {},
                    history_db = self.args.historydb,
                    options = self.args.options
                    )
            y = float(result[0][0])

            self.X.append(x)
            self.Y.append(y)
            state = 'OK'
            self.cpt += 1
            print(t, x, y, state, self.args.test_limit, self.cpt)
            # sys.stdout.flush()
        else:
            y = float("Inf")
            state = 'ERROR'
#            self.args.test_limit = min(self.args.test_limit + 1, 1000) #XXX Hack
            self.args.test_limit = self.args.test_limit + 1 #XXX Hack
        t2 = time.time_ns()
        self.timefun=self.timefun+(t2-t1)/1e9

        return opentuner.Result(time=y, state=state)

    def save_final_config(self, configuration):

        """called at the end of tuning"""

        self.Xopt = np.array([configuration.data[p] for p in self.args.tp.parameter_space.dimension_names])
        self.Yopt = min(self.Y)

####################################################################################################

def OpenTuner(T, NS, tp : TuningProblem, computer : Computer, historydb : HistoryDB = None, options : Options = None, run_id="OpenTuner", niter=1, technique=None):

    # Initialize

    args = argparse.Namespace()

    args.bail_threshold            = 500
    args.database                  = None
    args.display_frequency         = 10
    args.generate_bandit_technique = False
    args.label                     = run_id
    args.list_techniques           = False
    args.machine_class             = None
    args.no_dups                   = True
    args.parallel_compile          = False
    args.parallelism               = 1
    args.pipelining                = 0
    args.print_params              = False
    args.print_search_space_size   = False
    args.quiet                     = True
    args.results_log               = None
    args.results_log_details       = None
    args.seed_configuration        = []
    args.stop_after                = None
    args.technique                 = technique
    args.test_limit                = NS

    args.niter   = niter
    args.tp  = tp
    args.problem = Problem(tp, driverabspath=None, models_update=None)
    if (historydb is None):
        historydb = HistoryDB()
    args.historydb = historydb
    if (options is None):
        options = Options()
    args.options = options
    args.computer  = computer

    X = []
    Y = []
    data = Data(tp)
    # Xopt = []
    # Yopt = []

    # Tune
    stats = {
        "time_total": 0,
        "time_fun": 0
    }

    timefun=0
    t1 = time.time_ns()
    print("Start OpenTuner")
    for i in range(len(T)):

        args.t = T[i]
        args.test_limit = NS #XXX Hack
        a = OpenTunerInterface(args)
        b = opentuner.tuningrunmain.TuningRunMain(a, args)
        b.main()

        X.append(a.X)
        tmp = np.array(a.Y).reshape((len(a.Y), 1))
        Y.append(tmp)
        timefun=timefun+a.timefun
        # Xopt.append(a.Xopt)
        # Yopt.append(a.Yopt)

    print("End OpenTuner")

    t2 = time.time_ns()
    stats['time_total'] = (t2-t1)/1e9
    stats['time_fun'] = timefun

    data.I=T
    data.P=X
    data.O=Y
    # Finalize

    return (data, stats)

