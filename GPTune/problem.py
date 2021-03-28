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
import copy

class Problem(object):

    def __init__(self, tp : TuningProblem, driverabspath=None, models_update=None):

        self.name = tp.name

        self.IS = tp.input_space
        self.PS = tp.parameter_space
        self.OS = tp.output_space

        self.objectives   = tp.objective
        self.driverabspath   = driverabspath
        self.models_update   = models_update
        self.constraints = tp.constraints
        self.models      = tp.models
        self.constants      = tp.constants        

    @property
    def DI(self):

        return len(self.IS)

    @property
    def DP(self):

        return len(self.PS)

    @property
    def DO(self):

        return len(self.OS)

