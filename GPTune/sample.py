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
import numpy as np
import math
import skopt.space
from skopt.space import *
import time

from autotune.space import Space

class Sample(abc.ABC):

    @abc.abstractmethod
    def sample(self, n_samples : int, space : Space, **kwargs):

        raise Exception("Abstract method")

    def sample_constrained(self, n_samples : int, space : Space, check_constraints : Callable = None, check_constraints_kwargs : dict = {}, **kwargs):

        if (check_constraints is None):
            S = self.sample(n_samples, space)

        else:

            if ('sample_max_iter' in kwargs):
                sample_max_iter = kwargs['sample_max_iter']
            else:
                if ('options' in kwargs):
                    sample_max_iter = kwargs['options']['sample_max_iter']
                else:
                    sample_max_iter = 1

            S = []
            cpt = 0
            n_itr = 0
            while ((cpt < n_samples) and (n_itr < sample_max_iter)):
                # t1 = time.time_ns()
                S2 = self.sample(n_samples, space, kwargs=kwargs)
                # t2 = time.time_ns()
                # print('sample_para:',(t2-t1)/1e9)

                for s_norm in S2:
                    # print("jiji",s_norm)
                    s_orig = space.inverse_transform(np.array(s_norm, ndmin=2))[0]
                    kwargs2 = {d.name: s_orig[i] for (i, d) in enumerate(space)}
                    # print("dfdfdfdfd",kwargs2)
                    kwargs2.update(check_constraints_kwargs)
                    if (check_constraints(kwargs2)):
                        S.append(s_norm)
                        cpt += 1
                        if (cpt >= n_samples):
                            break
                # print('input',S,space[0],isinstance(space[0], Categorical))

                n_itr += 1
                if(n_itr%1000==0 and n_itr>=1000):
                    print('n_itr',n_itr,'still trying generating constrained samples...')


            if (cpt < n_samples):
                raise Exception("Only %d valid samples were generated while %d were requested.\
                        The constraints might be too hard to satisfy.\
                        Consider increasing 'sample_max_iter', or, provide a user-defined sampling method."%(len(S), n_samples))
        # print('reqi',S,'nsample',n_samples,sample_max_iter,space)
        S = np.array(S[0:n_samples]).reshape((n_samples, len(space)))

        return S

    def sample_inputs(self, n_samples : int, IS : Space, check_constraints : Callable = None, check_constraints_kwargs : dict = {}, **kwargs):

        return self.sample_constrained(n_samples, IS, check_constraints = check_constraints, check_constraints_kwargs = check_constraints_kwargs, **kwargs)

    def sample_parameters(self, n_samples : int, I : np.ndarray, IS : Space, PS : Space, check_constraints : Callable = None, check_constraints_kwargs : dict = {}, **kwargs):



        P = []
        for t in I:
            # print('before inverse_transform:',np.array(t, ndmin=2))
            I_orig = IS.inverse_transform(np.array(t, ndmin=2))[0]
            # I_orig = t
            # print('after inverse_transform I_orig:',I_orig)
            kwargs2 = {d.name: I_orig[i] for (i, d) in enumerate(IS)}
            kwargs2.update(check_constraints_kwargs)
            xs = self.sample_constrained(n_samples, PS, check_constraints = check_constraints, check_constraints_kwargs = kwargs2, **kwargs)
            P.append(xs)



        return P


import lhsmdu

class SampleLHSMDU(Sample):

    def __init__(self):

        super().__init__()

        self.cached_n_samples = None
        self.cached_space     = None
        self.cached_algo      = None

    def sample(self, n_samples : int, space : Space, **kwargs):

        kwargs = kwargs['kwargs']

        if (self.cached_n_samples is not None and self.cached_n_samples == n_samples and self.cached_space is not None and space == self.cached_space and self.cached_algo is not None and self.cached_algo == kwargs['sample_algo']):

            #lhs = lhsmdu.resample() # YC: this can fall into too clustered samples if there are many constraints
            lhs = lhsmdu.sample(len(space), n_samples)

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


import openturns as ot

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

        if (self.cached_space is not None and space == self.cached_space):
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

