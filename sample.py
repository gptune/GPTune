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

class Sample(abc.ABC):

    @abstractmethod
    def sample(self, n_samples : int, space : Space, **kwargs):

        raise Exception("Abstract method")

    def sample_constrained(self, n_samples : int, space : Space, check_constraints = None : Callable, check_constraints_kwargs = {} : Dict, **kwargs):

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
                S2 = self.sample(n_samples, space)
                for s_norm in S2:
                    s_orig = space.inverse_transform(np.array([s_norm], ndim=2))
                    kwargs2 = {d.name: s_orig[i] for (i, d) in enumerate(space)}
                    kwargs2.update(check_constraints_kwargs)
                    if (check_constraints(kwargs2)):
                        S.append(s_orig)
                        cpt += 1
                        if (cpt >= n_samples):
                            break
                n_itr += 1

            if (cpt < n_samples):
                raise Exception("Only %d valid samples were generated while %d were requested.\
                        The constraints might be too hard to satisfy.\
                        Consider increasing 'sample_max_iter', or, provide a user-defined sampling method."%(len(S), n_samples))

        S = np.array(S[0:n_samples]).reshape((n_samples, space.n_dim))

        return S

    def sample_inputs(self, n_samples : int, IS : Space, check_constraints = None : Callable, check_constraints_kwargs = {} : Dict, **kwargs):

        return self.sample_constrained(n_samples, IS, check_constraints = check_constraints, check_constraints_kwargs = check_constraints_kwargs, **kwargs)

    def sample_parameters(self, n_samples : int, T : np.ndarray, IS : Space, PS : Space, check_constraints = None : Callable, check_constraints_kwargs = {} : Dict, **kwargs):

        X = []
        for t in T:
            t_orig = IS.inverse_transform(np.array([t], ndim=2))
            kwargs2 = {d.name: t_orig[i] for (i, d) in enumerate(IS)}
            kwargs2.update(check_constraints_kwargs)
            xs = self.sample_constrained(n_samples, PS, check_constraints = check_constraints, check_constraints_kwargs = kwargs2, **kwargs)
            X.append(xs)

        return X

class SampleOpenTURNS(Sample):

    """
    XXX: This class, together with the underlying OpenTURNS only works on Intel-based CPUs.
    The reason is that OpenTURNS requires the Intel 'Thread Building Block' library to compile and execute.
    """

    import openturns as ot

    def sample(self, n_samples : int, space : Space, **kwargs):

        if (space == self.cached_space):
            distribution = self.cached_distribution
        else:
            distributions = [ot.Uniform(*d.transformed_bounds) for d in space)]
            distribution  = ot.ComposedDistribution(distributions)
            # Caching space and distribution to speed-up future samplings, especially if invoked by the sample_constrained method.
            self.cached_space = space
            self.cached_distribution = distribution 

        lhs = ot.LHSExperiment(distribution, n_samples)
        lhs.setAlwaysShuffle(True) # randomized

        S = lhs.generate()
        S.reshape((n_samples, space.n_dim))

        return S

