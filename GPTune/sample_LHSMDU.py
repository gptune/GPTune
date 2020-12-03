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


import abc
from typing import Callable
import math
import numpy as np
import time

import skopt.space
from skopt.space import *
from autotune.space import Space

import lhsmdu

from sample import Sample


class SampleLHSMDU(Sample):

    def __init__(self):

        super().__init__()

        self.cached_n_samples = None
        self.cached_space     = None
        self.cached_algo      = None

    def sample(self, n_samples : int, space : Space, **kwargs):

        kwargs = kwargs['kwargs']

        if (self.cached_n_samples is not None and self.cached_n_samples == n_samples and self.cached_space is not None and space == self.cached_space and self.cached_algo is not None and self.cached_algo == kwargs['sample_algo']):

            lhs = lhsmdu.resample()

        else:

            self.cached_n_samples = n_samples
            self.cached_space     = space
            self.cached_algo      = kwargs['sample_algo']

            if (kwargs['sample_algo'] == 'LHS-MDU'):
                lhs = lhsmdu.sample(len(space), n_samples)
            elif (kwargs['sample_algo'] == 'MCS'):
                lhs = lhsmdu.createRandomStandardUniformMatrix(len(space), n_samples)
            else:
                raise Excepetion(f"Unknown algorithm {kwargs['sample_algo']}")

        lhs = np.array(list(zip(*[np.array(lhs[k])[0] for k in range(len(lhs))])))
        # print(lhs,'normalized',n_samples)

        return lhs

