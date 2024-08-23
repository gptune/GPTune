import numpy as np

class MCMC_MetropolisHastings:
    def __init__(self, target_prob, proposal_prob=None, proposal_sample=None, ndim=1):
        self.p = target_prob
        self.q = proposal_prob
        self.q_sample = proposal_sample
        self.ndim = ndim  # Number of dimensions (hyperparameters)
        self.chains_data = np.empty((0, 0, self.ndim))  # Initialize with zero dimensions for samples, chains, ndim
        self.log_probs_data = np.empty((0, 0))  # Initialize with zero dimensions for samples, chains

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
        return thinned_log_probs
        if flat:
            # Reshape to combine chains and samples into a single dimension
            thinned_log_probs = thinned_log_probs.reshape(-1, self.ndim)


    def metropolis_hastings(self, x_init, iterations):
        x = x_init
        samples = np.zeros((iterations, self.ndim))  # Preallocate array for samples
        log_probs = np.zeros(iterations)  # Preallocate array for log probabilities

        for i in range(iterations):
            x_candidate = self.q_sample(x)
            accept_prob = min(1, (self.p(x_candidate) * self.q(x, x_candidate)) /
                                 (self.p(x) * self.q(x_candidate, x)))
            if np.random.rand() < accept_prob:
                x = x_candidate
            
            samples[i] = x  # Store the sample (now an ndim vector)
            log_probs[i] = np.log(self.p(x))  # Store the log probability
        
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
            self.chains_data = np.concatenate((self.chains_data, new_chains), axis=1)  # Concatenate along chain axis
            self.log_probs_data = np.concatenate((self.log_probs_data, new_log_probs), axis=1)  # Concatenate along chain axis

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
        
        # print(var_estimate[2],mean_within_chain_var[2],between_chain_var[2],nsteps,chain_means.shape)
        # Calculate the Gelman-Rubin statistic
        gelman_rubin_stat = np.sqrt(var_estimate / mean_within_chain_var)
        
        return gelman_rubin_stat

# Example usage with 2D problem:

# Define a 2D target distribution: Gaussian (simplified posterior for demonstration)
def p(x):
    # Assume x is a 2D vector [x1, x2]
    return np.exp(-0.5 * (x[0]**2 + x[1]**2)) / (2 * np.pi)

# Symmetric proposal distribution: normal centered at current position
def q(x, x_prime):
    # Multivariate Gaussian with identity covariance
    return np.exp(-0.5 * np.sum((x_prime - x)**2)) / (2 * np.pi)

# Proposal sampling function
def q_sample(x):
    return np.random.multivariate_normal(x, np.eye(2) * 0.1)

# Instantiate and use the class
mcmc = MCMC_MetropolisHastings(p, q, q_sample, ndim=2)
initial_positions = np.random.randn(4, 2)  # 4 chains, random initial positions in 2D space
mcmc.run_mcmc(initial_positions, 1000)
current_samples = mcmc.get_chain(discard=100, thin=1, flat=False)
R_hat = mcmc.gelman_rubin(current_samples)
print("Gelman-Rubin Statistic (R_hat):", R_hat)
