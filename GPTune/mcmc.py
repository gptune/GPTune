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
# import copy
# from typing import Collection, Tuple
import numpy as np
# from GPTune.problem import Problem
# from GPTune.computer import Computer
# from GPTune.data import Data

# import scipy.optimize as op
# import emcee
# from scipy.stats import truncnorm, gamma, invgamma, norm

import math
import time
from scipy.linalg import cho_solve, solve_triangular



class MCMCSampler_MetropolisHastings:
    """
    Adaptive random-walk Metropolis (Andrieu and Thoms 2008, Algorithm 4). The proposal is
    x + N(0, lambda*C) reflected into the bounds; the reflected Gaussian random walk is symmetric,
    so the acceptance ratio is the ratio of target densities, and no proposal is wasted at the
    bounds (where the optimal hyperparameters often are). The global scale lambda is tuned toward an
    acceptance rate of 0.234 and C follows the covariance of the chain, both by stochastic
    approximation with gain (t+1)^-0.6, which forgets the initial transient of the chain. The
    adaptation state of every chain persists across run_mcmc calls.
    """
    target_acceptance = 0.234

    def __init__(self, target_prob, bounds, ndim=1, nchain=1, initial_step=0.1, on_accept=None):
        self.p = target_prob # Target distribution
        self.on_accept = on_accept # called with (x, log probability) for the first state of every chain and every accepted proposal
        self.nchain = nchain
        self.bounds = None if bounds is None else np.array(bounds, dtype=float)
        self.ndim = ndim  # Number of dimensions (hyperparameters)
        self.chains_data = np.empty((0, 0, self.ndim))  # Initialize with zero dimensions for samples, chains, ndim
        self.log_probs_data = np.empty((0, 0))  # Initialize with zero dimensions for samples, chains
        self.initial_step = initial_step # standard deviation of the first proposals in every direction
        self.chain_state = {} # chain index -> current position, log probability and adaptation state

    def reflect(self, x):
        if self.bounds is None:
            return x
        lower, width = self.bounds[:, 0], self.bounds[:, 1] - self.bounds[:, 0]
        y = np.mod(x - lower, 2 * width)
        return lower + np.where(y > width, 2 * width - y, y)

    def cholesky(self, covariance):
        # the adapted covariance can be numerically singular in directions the chain has not moved in
        jitter = 1e-12 * max(np.trace(covariance) / self.ndim, 1e-12)
        while True:
            try:
                return np.linalg.cholesky(covariance + np.eye(self.ndim) * jitter)
            except np.linalg.LinAlgError:
                jitter *= 100

    def get_last_sample(self):
        from collections import namedtuple
        if self.chains_data.size == 0:
            raise ValueError("No data available. Please run run_mcmc first.")
        Sample = namedtuple('Sample', ['log_prob', 'coords'])
        return Sample(log_prob=self.log_probs_data[-1], coords=self.chains_data[-1])

    def get_chain(self, discard=0, thin=1, flat=False):
        if self.chains_data.size == 0:
            raise ValueError("No chains available. Please run run_mcmc first.")
        
        # Discard the first 'discard' samples and thin the remaining samples
        thinned_chains = self.chains_data[discard::thin]  # Apply discard and thinning
        if flat:
            # Reshape to combine chains and samples into a single dimension
            thinned_chains = thinned_chains.reshape(-1, self.ndim)
        return thinned_chains

    def get_log_prob(self, discard=0, thin=1, flat=False):
        if self.log_probs_data.size == 0:
            raise ValueError("No log_probs available. Please run run_mcmc first.")
        
        # Discard the first 'discard' samples and thin the remaining samples
        thinned_log_probs = self.log_probs_data[discard::thin]  # Apply discard and thinning
        if flat:
            # Reshape to combine chains and samples into a single dimension
            thinned_log_probs = thinned_log_probs.reshape(-1, 1)
        return thinned_log_probs

    def run_chain(self, chain, x_init, iterations):
        s = self.chain_state.get(chain)
        if s is None: # first call for this chain: start at x_init
            x = np.array(x_init, dtype=float)
            s = {'x': x, 'px': self.p(x, self.bounds), 't': 0, 'mean': x.copy(),
                 'cov': np.eye(self.ndim) * self.initial_step**2, 'log_lambda': 0.0}
            self.chain_state[chain] = s
            if self.on_accept is not None:
                self.on_accept(s['x'], s['px'])
        samples = np.zeros((iterations, self.ndim))  # Preallocate array for samples
        log_probs = np.zeros(iterations)  # Preallocate array for log probabilities

        for i in range(iterations):
            s['t'] += 1
            x_candidate = self.reflect(s['x'] + np.exp(0.5 * s['log_lambda']) * self.cholesky(s['cov']) @ np.random.standard_normal(self.ndim))
            px_candidate = self.p(x_candidate, self.bounds)
            accept_prob = 0.0 # a failed likelihood evaluation counts as a rejection
            if np.isfinite(px_candidate):
                accept_prob = np.exp(min(0.0, px_candidate - s['px']))
            if np.random.rand() < accept_prob:
                s['x'] = x_candidate
                s['px'] = px_candidate
                s['accepted'] = s.get('accepted', 0) + 1
                if self.on_accept is not None:
                    self.on_accept(s['x'], s['px'])
            samples[i] = s['x']
            log_probs[i] = s['px']

            # stochastic approximation of the proposal scale, and of the mean and covariance of the chain
            gamma = (s['t'] + 1) ** -0.6
            s['log_lambda'] += gamma * (accept_prob - self.target_acceptance)
            delta = s['x'] - s['mean']
            s['mean'] += gamma * delta
            s['cov'] += gamma * (np.outer(delta, delta) - s['cov'])

        return samples, log_probs  # Return arrays directly

    def chain_summary(self, s):
        """Current proposal scale and log probability of a chain."""
        return np.exp(0.5 * s['log_lambda']), s['px']

    def run_mcmc(self, initial_positions, iterations, deadline=None):
        """
        Advance the chains by `iterations` steps, taking turns step by step so that they progress
        together in time. With a deadline (a time.time() value), stop after the first round of steps
        that ends past it (at least one round is done). Returns the number of steps done per chain.
        """
        num_chains, ndim = initial_positions.shape
        new_chains = []
        new_log_probs = []
        start = {i: (self.chain_state[i]['t'] if i in self.chain_state else 0, self.chain_state[i].get('accepted', 0) if i in self.chain_state else 0)
                 for i in range(num_chains)}
        for step in range(iterations):
            if deadline is not None and step > 0 and time.time() > deadline:
                break
            steps = [self.run_chain(i, initial_positions[i], 1) for i in range(num_chains)]
            new_chains.append([sample[0] for sample, _ in steps])
            new_log_probs.append([log_prob[0] for _, log_prob in steps])

        for i in range(num_chains):
            s = self.chain_state[i]
            n = s['t'] - start[i][0]
            scale, log_p = self.chain_summary(s)
            print("%s chain %d: steps %d-%d, acceptance rate %.2f, step size %.3g, log target %.8e"
                  % (type(self).__name__.replace('MCMCSampler_', ''), i, start[i][0] + 1, s['t'],
                     (s.get('accepted', 0) - start[i][1]) / max(n, 1), scale, log_p))

        # Append new chains and log probs to existing ones
        new_chains = np.array(new_chains).reshape((-1, num_chains, ndim))  # Shape (iterations, num_chains, ndim)
        new_log_probs = np.array(new_log_probs).reshape((-1, num_chains))  # Shape (iterations, num_chains)

        if self.chains_data.size == 0:
            self.chains_data = new_chains
            self.log_probs_data = new_log_probs
        else:
            self.chains_data = np.concatenate((self.chains_data, new_chains), axis=0)  # Concatenate along iteration axis
            self.log_probs_data = np.concatenate((self.log_probs_data, new_log_probs), axis=0)  # Concatenate along iteration axis
        return len(new_chains)


class MCMCSampler_MALA(MCMCSampler_MetropolisHastings):
    """
    Metropolis-adjusted Langevin algorithm preconditioned with the Fisher information. The chains
    move in unbounded coordinates u, theta = lower + width*sigmoid(u), and target the density of u,
    pi(theta(u)) |dtheta/du|, so no proposal leaves the bounds and parameters whose best values lie
    on a bound can approach it (rejecting proposals outside the bounds stalls the chain there). The
    proposal is u' = u + (eps^2/2) M g(u) + eps M^(1/2) xi, with g the gradient of the log target of u
    and M^-1 = P = J F J + diag(2 s(1-s)) + I/max_step^2, where F is the Fisher information matrix of
    theta, J = dtheta/du, s = sigmoid(u) (2 s(1-s) is the curvature of the Jacobian term) and
    max_step limits the steps of u in directions the data hardly determine.
    - fisher_interval = 1: M = M(u) is computed at every proposal and the Metropolis-Hastings ratio
      uses the proposal densities N(u'; u + (eps^2/2) M(u) g(u), eps^2 M(u)) and
      N(u; u' + (eps^2/2) M(u') g(u'), eps^2 M(u')), log determinants included (simplified manifold
      MALA), so the chain targets the exact distribution. One Fisher matrix per step.
    - fisher_interval > 1: M is recomputed at the chain's current position every fisher_interval
      steps and used in both directions. Every stretch of steps with a fixed M leaves the target
      invariant, but choosing M from the current state biases the sampled distribution slightly (a
      few percent on bounded test targets, none on Gaussian ones).
    The step size eps is tuned toward an acceptance rate of 0.574. The samples and log probabilities
    kept are those of theta.
    """
    target_acceptance = 0.574
    max_step = 3.0

    def __init__(self, target_prob, bounds, ndim=1, nchain=1, grad_prob=None, fisher_prob=None, step_size=1.0, fisher_interval=1, on_accept=None):
        super().__init__(target_prob, bounds, ndim=ndim, nchain=nchain, on_accept=on_accept)
        if self.bounds is None:
            raise Exception("The MALA sampler needs bounds for all parameters")
        self.grad = grad_prob # gradient of the log target of theta
        self.fisher = fisher_prob # Fisher information matrix of theta
        self.step_size = step_size
        self.fisher_interval = fisher_interval
        self.lower = self.bounds[:, 0]
        self.width = self.bounds[:, 1] - self.bounds[:, 0]

    def to_u(self, theta):
        f = np.clip((np.asarray(theta, dtype=float) - self.lower) / self.width, 1e-12, 1 - 1e-12)
        return np.log(f) - np.log1p(-f)

    def evaluate(self, u):
        """theta(u), log target of theta, log target of u and its gradient"""
        sig = 0.5 * (1 + np.tanh(0.5 * u)) # sigmoid
        theta = np.clip(self.lower + self.width * sig, self.bounds[:, 0], self.bounds[:, 1]) # no rounding past the bounds
        log_p = self.p(theta, self.bounds)
        if not np.isfinite(log_p):
            return theta, log_p, log_p, None
        log_jacobian = np.sum(np.log(self.width) - np.logaddexp(0, -u) - np.logaddexp(0, u))
        g = self.width * sig * (1 - sig) * self.grad(theta) + (1 - 2 * sig)
        return theta, log_p, log_p + log_jacobian, g

    def preconditioner(self, u, previous=None):
        """
        The precision P of the proposal and its Cholesky factor L (P = L L^T). The proposal steps
        L^-T xi and the drift P^-1 g are computed from L, so they always match the proposal density
        used in the acceptance ratio, however ill-conditioned P is (any jitter added to factor P is
        part of P).
        """
        sig = 0.5 * (1 + np.tanh(0.5 * u))
        J = self.width * sig * (1 - sig)
        F = self.fisher(self.lower + self.width * sig)
        if not np.all(np.isfinite(F)):
            if previous is not None:
                return previous
            F = np.zeros((self.ndim, self.ndim))
        w, V = np.linalg.eigh(J[:, None] * F * J[None, :]) # the probe estimate of F is not guaranteed to be positive semidefinite
        P = (V * np.maximum(w, 0.0)) @ V.T + np.diag(2 * sig * (1 - sig) + 1 / self.max_step**2)
        P = 0.5 * (P + P.T)
        jitter = 0.0
        while True:
            try:
                P_jittered = P + np.eye(self.ndim) * jitter
                return P_jittered, np.linalg.cholesky(P_jittered)
            except np.linalg.LinAlgError:
                jitter = max(jitter * 100, 1e-12 * np.trace(P) / self.ndim)

    def run_chain(self, chain, x_init, iterations):
        position_dependent = self.fisher_interval == 1
        s = self.chain_state.get(chain)
        if s is None: # first call for this chain: start at x_init
            u = self.to_u(x_init)
            theta, log_p, log_pu, g = self.evaluate(u)
            s = {'u': u, 'theta': theta, 'log_p': log_p, 'log_pu': log_pu, 'g': g, 't': 0, 'log_eps': np.log(self.step_size), 'pre': None}
            if position_dependent:
                s['pre'] = self.preconditioner(u)
            self.chain_state[chain] = s
            if self.on_accept is not None:
                self.on_accept(theta, log_p)
        samples = np.zeros((iterations, self.ndim))  # Preallocate array for samples
        log_probs = np.zeros(iterations)  # Preallocate array for log probabilities

        for i in range(iterations):
            if not position_dependent and s['t'] % self.fisher_interval == 0:
                s['pre'] = self.preconditioner(s['u'], s['pre'])
            P, L = s['pre']
            s['t'] += 1
            eps = np.exp(s['log_eps'])
            mean_forward = s['u'] + 0.5 * eps**2 * cho_solve((L, True), s['g'])
            u_candidate = mean_forward + eps * solve_triangular(L.T, np.random.standard_normal(self.ndim), lower=False)

            accept_prob = 0.0 # a failed likelihood evaluation is rejected
            theta, log_p, log_pu, g = self.evaluate(u_candidate)
            if np.isfinite(log_pu):
                if position_dependent:
                    # the Fisher information at the candidate, whose factorization is the current one
                    pre_candidate = self.preconditioner(u_candidate)
                    P_back, L_back = pre_candidate
                    log_det_ratio = np.sum(np.log(np.diag(L_back))) - np.sum(np.log(np.diag(L))) # (log det P_back - log det P)/2
                else:
                    P_back, L_back, log_det_ratio = P, L, 0.0
                mean_backward = u_candidate + 0.5 * eps**2 * cho_solve((L_back, True), g)
                d_forward = u_candidate - mean_forward
                d_backward = s['u'] - mean_backward
                log_accept = (log_pu - s['log_pu'] + log_det_ratio
                              - 0.5 / eps**2 * (d_backward @ P_back @ d_backward - d_forward @ P @ d_forward))
                if np.isfinite(log_accept):
                    accept_prob = np.exp(min(0.0, log_accept))
                if np.random.rand() < accept_prob:
                    s.update(u=u_candidate, theta=theta, log_p=log_p, log_pu=log_pu, g=g)
                    if position_dependent:
                        s['pre'] = pre_candidate
                    s['accepted'] = s.get('accepted', 0) + 1
                    if self.on_accept is not None:
                        self.on_accept(theta, log_p)
            samples[i] = s['theta']
            log_probs[i] = s['log_p']

            s['log_eps'] += (s['t'] + 1) ** -0.6 * (accept_prob - self.target_acceptance)

        return samples, log_probs  # Return arrays directly

    def chain_summary(self, s):
        return np.exp(s['log_eps']), s['log_p']


class MCMC:
    def __init__(self, target_prob, bounds=None, ndim=1, nchain=1, mcmcsampler='MetropolisHastings', **sampler_options):
        if(mcmcsampler == 'MetropolisHastings'):
            self.sampler = MCMCSampler_MetropolisHastings(target_prob, bounds, ndim=ndim, nchain=nchain, **sampler_options)
        elif(mcmcsampler == 'MALA'):
            self.sampler = MCMCSampler_MALA(target_prob, bounds, ndim=ndim, nchain=nchain, **sampler_options)
        elif(mcmcsampler == 'Ensemble_emcee'):
            import emcee
            self.sampler = emcee.EnsembleSampler(nchain, ndim, target_prob)  
        else:
            raise Exception("MCMC sampler %s is not implemented"%(mcmcsampler))

    def gelman_rubin(self,samples):
        """
        Compute the Gelman-Rubin diagnostic statistic (R-hat) for convergence.
        
        Parameters:
        samples (np.ndarray): MCMC samples of shape (nsteps, nwalkers, ndim)
        
        Returns:
        np.ndarray: Gelman-Rubin statistic for each dimension
        """

        nsteps, nwalkers, ndim = samples.shape
        
        # Calculate the within-chain variance for each dimension
        within_chain_var = np.var(samples, axis=0, ddof=1)
        
        # Calculate the mean of the samples for each step and dimension
        chain_means = np.mean(samples, axis=0)
        
        # Calculate the mean of the means for each dimension
        mean_of_means = np.mean(chain_means, axis=0)
        
        # Calculate the between-chain variance for each dimension: B = nsteps/(nwalkers-1) * sum_j (mean_j - mean)^2
        between_chain_var = np.sum((chain_means - mean_of_means) ** 2, axis=0)*nsteps/(nwalkers-1)

        # Calculate the mean within-chain variance for each dimension
        mean_within_chain_var = np.mean(within_chain_var, axis=0)

        # Calculate the variance estimate
        var_estimate = ((nsteps - 1) / nsteps) * mean_within_chain_var + (1 / nsteps) * between_chain_var

        # print(var_estimate,mean_within_chain_var,between_chain_var,nsteps,chain_means.shape,'gelman_rubin_stat')
        # Calculate the Gelman-Rubin statistic (infinite for a chain that has not moved in some dimension)
        with np.errstate(divide='ignore', invalid='ignore'):
            gelman_rubin_stat = np.sqrt(var_estimate / mean_within_chain_var)
        gelman_rubin_stat[~np.isfinite(gelman_rubin_stat)] = np.inf
        
        return gelman_rubin_stat

    def map_result(self, status, message):
        """The MAP sample of all chains, as the result of the sampling."""
        flat_samples = self.sampler.get_chain(discard=0, thin=1, flat=True)
        flat_log_posteriors = np.ravel(self.sampler.get_log_prob(discard=0, thin=1, flat=True))
        map_index = np.argmax(flat_log_posteriors)
        return type('Result', (object,), {'x': flat_samples[map_index], 'success': True, 'status': status, 'message': message,
                                          'fun': -float(flat_log_posteriors[map_index]), 'nfev': flat_samples.shape[0], 'nit': flat_samples.shape[0]})()

    def run_mcmc_with_convergence(self, initial_state, n_steps, discard=100, thin=1, check_interval=100, r_hat_threshold=1.5, verbose=False, max_time=None):
        """
        Run the chains for up to n_steps steps each, checking the Gelman-Rubin statistic every
        check_interval steps, and return the MAP sample once all R-hat < r_hat_threshold, after n_steps
        steps, or once the sampling has run for max_time seconds (if given; checked after every round
        of steps of the MetropolisHastings and MALA samplers, after every check_interval steps of emcee).
        """
        nwalkers, ndim = initial_state.shape
        deadline = None if max_time is None else time.time() + max_time

        for i in range(0, n_steps, check_interval):
            nsteps = min(check_interval, n_steps - i)
            if isinstance(self.sampler, MCMCSampler_MetropolisHastings):
                done = self.sampler.run_mcmc(initial_state, nsteps, deadline)
            else:
                self.sampler.run_mcmc(initial_state, nsteps)
                done = nsteps
            initial_state = self.sampler.get_last_sample().coords
            if deadline is not None and (done < nsteps or time.time() > deadline):
                if(verbose==True):
                    print(f"MCMC stopped after {i + done} steps: maximum sampling time {max_time} s reached")
                return self.map_result(2, 'Maximum sampling time reached')

            if i >= check_interval:
                current_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=False)
                if current_samples.shape[0] < 2:
                    continue
                r_hat = self.gelman_rubin(current_samples)
                if(verbose==True):
                    print(f"MCMC Step {i + nsteps}: R-hat = {r_hat}")
                if np.all(r_hat < r_hat_threshold):
                    return self.map_result(0, 'MCMC converged')

        return self.map_result(1, 'Maximum number of iterations reached')
