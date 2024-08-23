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
# from problem import Problem
# from computer import Computer
# from data import Data

# import scipy.optimize as op
# import emcee
# from scipy.stats import truncnorm, gamma, invgamma, norm

import math



class MCMCSampler_MetropolisHastings:
    def __init__(self, target_prob, ndim=1, nchain=1):
        self.p = target_prob # Target distribution
        self.nchain = nchain
        self.ndim = ndim  # Number of dimensions (hyperparameters)
        self.chains_data = np.empty((0, 0, self.ndim))  # Initialize with zero dimensions for samples, chains, ndim
        self.log_probs_data = np.empty((0, 0))  # Initialize with zero dimensions for samples, chains
        self.scale = 1.0  # Initial scale for the covariance matrix
        self.adaptation_interval = 10  # Adapt covariance every 100 iterations

    # Update the proposal distribution to include covariance
    def q(self, x, x_prime, covariance):
        det = np.linalg.det(covariance)
        norm_const = np.sqrt((2 * np.pi)**self.ndim * det)
        diff = x_prime - x
        return np.exp(-0.5 * diff.T @ np.linalg.inv(covariance) @ diff) / norm_const

    # Update the proposal sample function to use the dynamic covariance
    def q_sample(self, x, covariance):
        return np.random.multivariate_normal(x, covariance)


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

    def metropolis_hastings(self, x_init, iterations):
        x = x_init
        samples = np.zeros((iterations, self.ndim))  # Preallocate array for samples
        log_probs = np.zeros(iterations)  # Preallocate array for log probabilities
        covariance = np.eye(self.ndim) * self.scale  # Initial covariance
        px = self.p(x) 

        for i in range(iterations):
            x_candidate = self.q_sample(x, covariance)
            px_candiate = self.p(x_candidate)

            accept_prob = min(1, (np.exp(px_candiate) * self.q(x, x_candidate, covariance)) /
                                 (np.exp(px) * self.q(x_candidate, x, covariance)))
            if np.random.rand() < accept_prob:
                x = x_candidate
                log_probs[i] = px_candiate  # Store the log probability    
                px = px_candiate            
            else:
                log_probs[i] = px  # Store the log probability
            
            samples[i] = x  # Store the sample (now an ndim vector)            

            if (i + 1) % self.adaptation_interval == 0:
                # Adapt the covariance based on the sample variance
                covariance = np.cov(samples[:i+1].T) + np.eye(self.ndim) * 1e-6  # Adding a small value for numerical stability
        
        return samples, log_probs  # Return arrays directly

    def run_mcmc(self, initial_positions, iterations):
        new_chains = []
        new_log_probs = []
        num_chains, ndim = initial_positions.shape

        for i in range(num_chains):
            x_init = initial_positions[i]
            chain, log_probs = self.metropolis_hastings(x_init, iterations)
            new_chains.append(chain)
            new_log_probs.append(log_probs)

        # Append new chains and log probs to existing ones
        new_chains = np.array(new_chains).transpose((1, 0, 2))  # Shape (iterations, num_chains, ndim)
        new_log_probs = np.array(new_log_probs).transpose((1, 0))  # Shape (iterations, num_chains)

        if self.chains_data.size == 0:
            self.chains_data = new_chains
            self.log_probs_data = new_log_probs
        else:
            self.chains_data = np.concatenate((self.chains_data, new_chains), axis=0)  # Concatenate along iteration axis
            self.log_probs_data = np.concatenate((self.log_probs_data, new_log_probs), axis=0)  # Concatenate along iteration axis
            

class MCMC:
    def __init__(self, target_prob, ndim=1, nchain=1, mcmcsampler='MetropolisHastings'):
        if(mcmcsampler is 'MetropolisHastings'):
            self.sampler = MCMCSampler_MetropolisHastings(target_prob, ndim=ndim, nchain=nchain)
        elif(mcmcsampler is 'Ensemble_emcee'):
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
        
        # Calculate the between-chain variance for each dimension
        between_chain_var = np.mean((chain_means - mean_of_means) ** 2, axis=0)*nsteps
        
        # Calculate the mean within-chain variance for each dimension
        mean_within_chain_var = np.mean(within_chain_var, axis=0)
        
        # Calculate the variance estimate
        var_estimate = ((nsteps - 1) / nsteps) * mean_within_chain_var + (1 / nsteps) * between_chain_var
        
        # print(var_estimate,mean_within_chain_var,between_chain_var,nsteps,chain_means.shape,'gelman_rubin_stat')
        # Calculate the Gelman-Rubin statistic
        gelman_rubin_stat = np.sqrt(var_estimate / mean_within_chain_var)
        
        return gelman_rubin_stat

    def run_mcmc_with_convergence(self, initial_state, n_steps, discard=100, thin=1, check_interval=100, r_hat_threshold=1.5,verbose=False):
        nwalkers, ndim = initial_state.shape
        
        for i in range(0, n_steps, check_interval):
            self.sampler.run_mcmc(initial_state, check_interval)
            initial_state = self.sampler.get_last_sample().coords
            current_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=False)
            current_log_prob = self.sampler.get_log_prob(discard=discard, thin=thin, flat=False)
            # print(self.sampler.get_log_prob(discard=0, thin=thin, flat=False),'really')
            # if current_samples.shape[0]>0:
            #     current_samples = sampler.get_chain(discard=0, thin=1, flat=False)
            # Print shapes for debugging
            # print(f"Iteration {i}: current_samples shape = {current_samples.shape}")
            # print(current_samples[:,:,2])
            
            end_index = i + check_interval
            if end_index > n_steps:
                end_index = n_steps
                
            # Check if the slice is valid
            if current_samples.shape[0] < (end_index - i):
                # print(f"Warning: Not enough samples to fill the required slice. Current samples shape: {current_samples.shape}")
                continue
            
            if i >= check_interval:
                r_hat = self.gelman_rubin(current_samples)
                if(verbose==True):
                    print(f"MCMC Step {i + check_interval}: R-hat = {r_hat}")                
                if np.all(r_hat < r_hat_threshold):
                    # print("MCMC Chains have converged.")
                    flat_samples = self.sampler.get_chain(discard=0, thin=1, flat=True)
                    flat_log_posteriors = self.sampler.get_log_prob(discard=0, thin=1, flat=True)
                    # ####### Get the best parameter estimate
                    # best_params = np.mean(flat_samples, axis=0)
                    # print("Mean of samples:", best_params)
                    # np.save('flat_log_posteriors.npy',flat_samples)
                    # np.save('flat_samples.npy',flat_log_posteriors)
                    map_index = np.argmax(flat_log_posteriors)
                    best_params = flat_samples[map_index]
                    # print("MAP sample:", best_params)
                    resopt = type('Result', (object,), {'x': best_params, 'success': True, 'status': 0, 'message': 'MCMC converged', 'fun': -flat_log_posteriors[map_index], 'nfev': end_index, 'nit': end_index})
                    return resopt



        # print("Reached maximum steps without full convergence.")
        flat_samples = self.sampler.get_chain(discard=0, thin=1, flat=True)
        flat_log_posteriors = self.sampler.get_log_prob(discard=0, thin=1, flat=True)
        # ####### Get the best parameter estimate
        # best_params = np.mean(flat_samples, axis=0)
        # print("Mean of samples:", best_params)
        map_index = np.argmax(flat_log_posteriors)
        best_params = flat_samples[map_index]
        # print("MAP sample:", best_params)
        resopt = type('Result', (object,), {'x': best_params, 'success': True, 'status': 1, 'message': 'Maximum number of iterations reached', 'fun': -flat_log_posteriors[map_index], 'nfev': flat_samples.shape[0], 'nit': flat_samples.shape[0]})()


        return resopt

