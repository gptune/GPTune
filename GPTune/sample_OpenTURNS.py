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

import openturns as ot

from sample import Sample


class SampleOpenTURNS(Sample):

    """
    XXX: This class, together with the underlying OpenTURNS only works on Intel and AMD CPUs.
    The reason is that OpenTURNS requires the Intel 'Thread Building Block' library to compile and execute.
    """

    def __init__(self):

        super().__init__()

        self.cached_space        = None
        self.cached_distribution = None

    def sample(self, n_samples : int, space : Space, **kwargs):

        if (self.cached_space is not None and space is self.cached_space):
            distribution = self.cached_distribution
        else:
            distributions = [ot.Uniform(*d.transformed_bounds) for d in space.dimensions]
            distribution  = ot.ComposedDistribution(distributions)
            # Caching space and distribution to speed-up future samplings, especially if invoked by the sample_constrained method.
            self.cached_space = space
            self.cached_distribution = distribution

        lhs = ot.LHSExperiment(distribution, n_samples)
        lhs.setAlwaysShuffle(True) # randomized

        S = lhs.generate()
        S = np.array(S)
        S.reshape((n_samples, len(space)))

        return S

