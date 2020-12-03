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


from autotune.problem import TuningProblem

from problem import Problem
from computer import Computer
from data import Data
from options import Options
from sample import *
from model import *
from search import *


class GPTune(object):

    def __init__(
            self,
            tuningproblem : TuningProblem,
            computer : Computer = None,
            data : Data = None,
            options : Options = None,
            driverabspath=None,
            models_update=None,
            **kwargs
            ):

        """
        tuningproblem: object defining the characteristics of the tuning (See file 'autotuner/autotuner/tuningproblem.py')
        computer     : object specifying the architectural characteristics of the computer to run on (See file 'GPTune/computer.py')
        data         : object containing the data of a previous tuning (See file 'GPTune/data.py')
        options      : object defining all the options that will define the behaviour of the tuner (See file 'GPTune/options.py')
        """

        self.problem  = Problem(tuningproblem,driverabspath=driverabspath,models_update=models_update)
        if (computer is None):
            computer = Computer()
        self.computer = computer
        if (data is None):
            data = Data(self.problem)
        self.data     = data
        if (options is None):
            options = Options()
        self.options  = options

        # Imported methods
        from mla  import MLA
        from tla1 import TLA1
        from tla2 import TLA2

