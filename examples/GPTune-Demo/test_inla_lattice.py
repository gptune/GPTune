#! /usr/bin/env python
"""
Checks of the INLA/SPDE Gaussian process of george (george/inla.py) on small 2D and 3D lattices, for
several smoothness values nu (exact models when nu + d/2 is an integer, approximations otherwise).

Without SuperLU_DIST workers (python test_inla_lattice.py):
  1. log|Q|, computed analytically from the 1D generalized eigenvalues, against scipy's sparse LU of Q,
     and Q^-1 applied with the 1D eigenvectors against scipy's sparse solve.
  2. The marginal variance and the correlations of the discretized field, far from the boundary,
     against the Matern model (variance sigma^2, length scales l_d, smoothness nu).
With SuperLU_DIST workers (test_GP_inla.sh runs python test_inla_lattice.py -superlu 1, which skips 1 and 2):
  3. log|Q_c| from SuperLU_DIST against scipy's sparse LU.
  4. The log-likelihood against the dense Gaussian log-likelihood of y, of covariance A Q^-1 A' + s^2 I.
  5. The predictive means, variances (Rao-Blackwellized estimates) and covariances against the dense
     posterior, at points on the lattice nodes and between them.
  6. The gradient of the log-likelihood and the Fisher information against central finite differences
     of the dense log-likelihood and covariance of y, with exact traces (identity probes of the
     observation space) and with the Hutchinson estimates of 64 probes.
"""

import argparse
import os
import time

import numpy as np
from scipy.sparse.linalg import splu, spsolve
from scipy.special import gamma, kv

import george
# use the INLA engine of the george source tree next to GPTune if the installed george does not have it
GEORGE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../george/src/george')
george.__path__.insert(0, GEORGE_SRC)
from george.inla import INLAGP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-superlu', type=int, default=0, help='Whether to run the checks needing SuperLU_DIST workers')
    return parser.parse_args()


def logdet_splu(M):
    """log|det M| from scipy's sparse LU (L has a unit diagonal)."""
    return np.sum(np.log(np.abs(splu(M.tocsc()).U.diagonal())))


def matern_correlation(r, nu):
    u = np.sqrt(2 * nu) * np.maximum(r, 1e-300)
    return np.where(r == 0, 1.0, 2 ** (1 - nu) / gamma(nu) * u ** nu * kv(nu, u))


def params_vector(noise, amplitude, lengthscales):
    return np.concatenate([[np.log(noise), np.log(amplitude)], 2 * np.log(lengthscales)])


def check_logdet(d, nu, rng):
    shape = {2: (15, 11), 3: (7, 6, 5)}[d]
    x = rng.uniform(0, 1, (60, d))
    m = INLAGP(d, shape=shape, bounds=[(0, 1)] * d, buffer=0.3, nu=nu)
    m.compute(x)
    print("check 1 (%dD, nu=%g, alpha=%g, lattice %s with buffer, %d nodes, %d nonzeros in Q): log|Q| analytic vs scipy LU"
          % (d, m.nu, m.alpha, "x".join(map(str, m._shape)), m._n, len(m._keys)))
    for _ in range(2):
        p = params_vector(0.01, rng.uniform(0.5, 3), rng.uniform(0.1, 0.4, d))
        analytic = m.logdet_prior(p)
        reference = logdet_splu(m._assemble(p, data_term=False))
        print("   lengthscales %s: %.10e vs %.10e, relative difference %.1e"
              % (np.round(np.sqrt(np.exp(p[2:])), 3), analytic, reference, abs(analytic - reference) / abs(reference)))
        m.set_parameter_vector(p)
        rhs = rng.standard_normal((m._n, 3))
        fast = m._prior_solve(rhs)
        reference = spsolve(m._assemble(p, data_term=False).tocsc(), rhs)
        print("   Q^-1 with the 1D eigenvectors vs scipy solve: relative difference %.1e"
              % (np.linalg.norm(fast - reference) / np.linalg.norm(reference)))


def check_marginal(d, nu):
    shape, lengthscales = {2: ((81, 81), np.array([0.15, 0.1])), 3: ((25, 25, 25), np.array([0.2, 0.25, 0.3]))}[d]
    sigma2 = 1.5
    x = np.full((1, d), 0.5)
    m = INLAGP(d, shape=shape, bounds=[(0, 1)] * d, buffer=2 * lengthscales.max(), nu=nu)
    m.compute(x)
    p = params_vector(0.01, sigma2, lengthscales)
    m.set_parameter_vector(p)
    start = time.time()
    center = m._A.indices[0]  # the node at the center
    column = m._prior_solve(np.eye(m._n)[:, center])  # the covariances of the center with all the nodes (Q^-1 checked by check 1)
    print("check 2 (%dD, nu=%g, alpha=%g (%s), lattice %s with buffer, %d nodes, l=%s, l/h=%s, %.1f s): Matern marginal variance and correlations"
          % (d, nu, m.alpha, "exact" if abs(m.alpha - round(m.alpha)) < 1e-12 else "approximation", "x".join(map(str, m._shape)),
             m._n, lengthscales, np.round(lengthscales / m.spacing, 1), time.time() - start))
    print("   variance at the center %.4f vs sigma^2 %.4f" % (column[center], sigma2))
    strides = [int(np.prod(m._shape[k + 1:])) for k in range(d)]
    for k in range(d):
        steps = np.array([1, 2, 4, 8])
        r = steps * m.spacing[k] / lengthscales[k]
        discrete = column[center + steps * strides[k]] / column[center]
        print("   dimension %d, lags %s (in length scales): correlations %s vs Matern %s"
              % (k, np.round(r, 2), np.round(discrete, 4), np.round(matern_correlation(r, nu), 4)))


def check_superlu(d, nu, rng):
    shape = {2: (15, 11), 3: (7, 6, 5)}[d]
    # half of the inputs on lattice nodes, half between them
    nodes_x = [np.linspace(0, 1, s) for s in shape]
    on_nodes = np.stack([rng.choice(z, 30) for z in nodes_x], axis=1)
    x = np.vstack([on_nodes, rng.uniform(0, 1, (30, d))])
    y = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + 0.05 * rng.standard_normal(len(x)) + 10.0
    lengthscales = rng.uniform(0.15, 0.35, d)
    m = INLAGP(d, noise_variance=0.01, amplitude=0.8, lengthscale=lengthscales, shape=shape,
               bounds=[(0, 1)] * d, buffer=0.3, nsamples=256, seed=1, nu=nu)
    m.compute(x)
    print("check 3-5 (%dD, nu=%g, alpha=%g, lattice %s with buffer, %d nodes, %d observations)"
          % (d, m.nu, m.alpha, "x".join(map(str, m._shape)), m._n, len(y)))
    xtest = np.vstack([np.stack([rng.choice(z, 20) for z in nodes_x], axis=1), rng.uniform(0, 1, (20, d))])
    for trial in range(2):
        p = params_vector(0.01 * (trial + 1), 0.8 / (trial + 1), lengthscales * (1 + 0.3 * trial))
        m.set_parameter_vector(p)
        ll = m.log_likelihood(y)

        # dense references
        s2 = np.exp(p[0])
        A = m._A.toarray()
        Q = m._assemble(p, data_term=False).toarray()
        Qc = m._assemble(p).toarray()
        r = y - y.mean()
        cov_y = A @ np.linalg.solve(Q, A.T) + s2 * np.eye(len(y))
        ll_dense = -0.5 * (np.linalg.slogdet(cov_y)[1] + r @ np.linalg.solve(cov_y, r) + len(y) * np.log(2 * np.pi))
        posterior = np.linalg.inv(Qc)
        xhat = posterior @ (A.T @ r / s2)
        At = np.zeros((len(xtest), m._n))
        nodes, weights, _ = m._interpolation(xtest)
        for i in range(len(xtest)):
            At[i, nodes[i]] = weights[i]
        mu_dense = At @ xhat + y.mean()
        cov_dense = At @ posterior @ At.T
        var_dense = np.diag(cov_dense)

        mu, var = m.predict(y, xtest, return_var=True)
        mu2, cov = m.predict(y, xtest, return_cov=True)
        err_var = np.abs(var - var_dense) / var_dense
        print("   trial %d: log|Q_c| SuperLU %.10e vs scipy %.10e" % (trial, m._logdet_qc, logdet_splu(m._qc)))
        print("            log-likelihood %.10e vs dense %.10e, relative difference %.1e" % (ll, ll_dense, abs(ll - ll_dense) / abs(ll_dense)))
        print("            predictive mean: max difference %.1e (max |mean| %.1e)" % (np.max(np.abs(mu - mu_dense)), np.max(np.abs(mu_dense))))
        print("            predictive variance (%d samples): relative error max %.3f, median %.3f; on nodes max %.3f, between nodes max %.3f"
              % (m.nsamples, err_var.max(), np.median(err_var), err_var[:20].max(), err_var[20:].max()))
        print("            predictive covariance (exact solves): max difference %.1e (max variance %.1e), mean max difference %.1e"
              % (np.max(np.abs(cov - cov_dense)), var_dense.max(), np.max(np.abs(mu2 - mu_dense))))


def dense_covariance(m, p):
    """The covariance A Q^-1 A' + s^2 I of the observations, dense."""
    A = m._A.toarray()
    return A @ np.linalg.solve(m._assemble(p, data_term=False).toarray(), A.T) + (np.exp(p[0]) + m._yerr2) * np.eye(A.shape[0])


def dense_loglik(m, p, y):
    r = y - y.mean()
    S = dense_covariance(m, p)
    return -0.5 * (np.linalg.slogdet(S)[1] + r @ np.linalg.solve(S, r) + len(y) * np.log(2 * np.pi))


def check_gradient(d, isotropic, nu, rng):
    shape = {2: (15, 11), 3: (7, 6, 5)}[d]
    nodes_x = [np.linspace(0, 1, s) for s in shape]
    # more observations than probes, so that 64 probes give Hutchinson estimates
    x = np.vstack([np.stack([rng.choice(z, 150) for z in nodes_x], axis=1), rng.uniform(0, 1, (150, d))])
    y = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + 0.05 * rng.standard_normal(len(x))
    lengthscales = rng.uniform(0.15, 0.35, 1 if isotropic else d)
    m = INLAGP(d, noise_variance=0.01, amplitude=0.8, lengthscale=lengthscales, isotropic=isotropic, shape=shape,
               bounds=[(0, 1)] * d, buffer=0.3, nu=nu)
    m.compute(x)
    p = m.get_parameter_vector()
    step = 1e-5
    shifts = np.eye(len(p)) * step
    grad_ref = np.array([(dense_loglik(m, p + e, y) - dense_loglik(m, p - e, y)) / (2 * step) for e in shifts])
    S_inv = np.linalg.inv(dense_covariance(m, p))
    dS = [(dense_covariance(m, p + e) - dense_covariance(m, p - e)) / (2 * step) for e in shifts]
    fisher_ref = np.array([[0.5 * np.trace(S_inv @ Di @ S_inv @ Dj) for Dj in dS] for Di in dS])
    print("check 6 (%dD%s, nu=%g, alpha=%g, %d nodes, %d observations): gradient and Fisher information vs finite differences of the dense likelihood"
          % (d, ", isotropic" if isotropic else "", m.nu, m.alpha, m._n, len(y)))
    print("   reference gradient %s" % np.array2string(grad_ref, precision=4))
    for nprobe in (10 ** 6, 64):
        m.nprobe = nprobe
        m.set_parameter_vector(p)
        start = time.time()
        grad = m.grad_log_likelihood(y)
        t_grad = time.time() - start
        start = time.time()
        fisher = m.fisher_information()
        t_fisher = time.time() - start
        label = "exact traces" if nprobe >= len(y) else "%d probes" % nprobe
        print("   %-12s gradient %s, relative error per component %s (%.2f s)"
              % (label, np.array2string(grad, precision=4), np.array2string(np.abs(grad - grad_ref) / np.abs(grad_ref), formatter={'float_kind': lambda v: '%.1e' % v}), t_grad))
        print("   %-12s Fisher information: relative error (Frobenius) %.1e (%.2f s)"
              % (label, np.linalg.norm(fisher - fisher_ref) / np.linalg.norm(fisher_ref), t_fisher))


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(2026)
    # (dimension, nu): alpha = nu + d/2 is an integer (exact model) or not (approximation)
    smoothness = [(2, 0.5), (2, 1.0), (2, 1.5), (2, 2.0), (2, 2.5), (3, 0.5), (3, 1.0), (3, 1.5)]
    if args.superlu == 0:
        for d, nu in smoothness:
            check_logdet(d, nu, rng)
        for d, nu in smoothness:
            check_marginal(d, nu)
    else:
        from pdbridge import superlu_factor, superlu_logdet, superlu_solve, superlu_terminate
        import george.solvers.basic
        for f in (superlu_factor, superlu_logdet, superlu_solve):
            setattr(george.solvers.basic, f.__name__, f)
        for d, nu in ((2, 1.0), (2, 0.5), (2, 2.0), (3, 0.5), (3, 1.5)):
            check_superlu(d, nu, rng)
        for d, isotropic, nu in ((2, False, 1.0), (2, True, 1.0), (2, False, 0.5), (2, False, 2.0), (3, False, 0.5), (3, False, 1.5)):
            check_gradient(d, isotropic, nu, rng)
