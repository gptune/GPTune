#! /usr/bin/env python
"""
Checks of the INLA/SPDE model on the sphere of george (george/inla.py, SphereINLAGP) against dense
references on small icosahedral meshes (levels 3 and 4: 642 and 2562 nodes):
  1. the mesh and the finite elements: node count, B'B = G, total area, Q = tau^2 L (C^-1 L)^(alpha-1)
     and F F' = Q (the square root of the samples), the barycentric interpolation
  2. the normalization: the mean marginal variance of the discrete field against sigma^2, and its
     correlations against the Matern correlation of the chordal distance (informative: the SPDE field
     on the sphere is not exactly Matern)
  3. log|Q| from the factorization of L against the dense log-determinant
  4. the log-likelihood, the predictive mean, variances and covariance against the dense Gaussian model
  5. the gradient and the Fisher information against finite differences of the dense likelihood
Without -superlu 1, the factorizations use scipy's sparse LU in place of the SuperLU_DIST bridge (the
same checks, on a login node); with it, the pdbridge workers must be running (test_GP_inla_sphere.sbatch).
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import splu
from scipy.special import gamma, kv

GPTUNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
try:
    import dPy_BPACK_wrapper  # noqa: F401 (imported by george.solvers.basic, not used here)
except ImportError:
    import types
    sys.modules['dPy_BPACK_wrapper'] = types.ModuleType('dPy_BPACK_wrapper')
import george
george.__path__.insert(0, os.path.join(GPTUNE_ROOT, 'george/src/george'))
import george.inla as inla
from george.inla import SphereINLAGP, _icosphere, _fem_sphere, _sphere_variance_sum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-superlu', type=int, default=0, help='1: factor with the SuperLU_DIST workers (pdbridge) instead of scipy')
    return parser.parse_args()


class ScipyBridge(object):
    """superlu_factor/logdet/solve of pdbridge with scipy's LU: one factorization held at a time."""
    lu = None

    @staticmethod
    def factor(KV, INT64=0, algo3d=0, verbosity=False):
        ScipyBridge.lu = splu(KV.tocsc())

    @staticmethod
    def logdet(verbosity=False):
        lu = ScipyBridge.lu
        diag = lu.U.diagonal()
        sign = np.prod(np.sign(diag)) * ScipyBridge.parity(lu.perm_r) * ScipyBridge.parity(lu.perm_c)
        return (1 if sign > 0 else -1), float(np.sum(np.log(np.abs(diag))))

    @staticmethod
    def solve(vec, verbosity=False):
        vec[...] = ScipyBridge.lu.solve(np.ascontiguousarray(vec))

    @staticmethod
    def parity(perm):
        seen = np.zeros(len(perm), dtype=bool)
        sign = 1
        for i in range(len(perm)):
            if not seen[i]:
                j, length = i, 0
                while not seen[j]:
                    seen[j] = True
                    j = perm[j]
                    length += 1
                if length % 2 == 0:
                    sign = -sign
        return sign


def matern_correlation(r, nu):
    u = np.sqrt(2 * nu) * np.maximum(r, 1e-300)
    return np.where(r == 0, 1.0, 2 ** (1 - nu) / gamma(nu) * u ** nu * kv(nu, u))


def random_sphere(rng, n):
    x = rng.standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1)[:, None]


def params_vector(noise, amplitude, lengthscale):
    return np.array([np.log(noise), np.log(amplitude), 2 * np.log(lengthscale)])


def dense_prior(m, p):
    return m._assemble(p, data_term=False).toarray()


def dense_covariance(m, p):
    A = m._A.toarray()
    return A @ np.linalg.solve(dense_prior(m, p), A.T) + (np.exp(p[0]) + m._yerr2) * np.eye(A.shape[0])


def dense_loglik(m, p, y):
    r = y - y.mean()
    S = dense_covariance(m, p)
    return -0.5 * (np.linalg.slogdet(S)[1] + r @ np.linalg.solve(S, r) + len(y) * np.log(2 * np.pi))


def check_mesh(level, nu, rng):
    v, t = _icosphere(level)
    mass, stiff, gradient = _fem_sphere(v, t)
    print("check 1 (level %d, nu=%g): %d nodes (expected %d), %d triangles; total area %.6f (sphere %.6f); "
          "max |B'B - G| %.1e; mass of the vertices with 5 neighbours %.4f, with 6 %.4f"
          % (level, nu, len(v), 10 * 4 ** level + 2, len(t), mass.sum(), 4 * np.pi,
             abs(gradient.T @ gradient - stiff).max(), mass[:12].mean(), mass[12:].mean()))
    m = SphereINLAGP(level=level, nu=nu, noise_variance=0.01, amplitude=0.8, lengthscale=0.3)
    x = np.vstack([v[rng.choice(len(v), 20, replace=False)], random_sphere(rng, 40)])
    m.compute(x)
    s2, sigma2, tau2, lam = m._hyperparameters()
    Q = dense_prior(m, m.get_parameter_vector())
    L = np.diag(mass) + lam[0] * stiff.toarray()
    Qref = tau2 * L.copy()
    for _ in range(m.alpha - 1):
        Qref = Qref @ np.diag(1 / mass) @ L
    F = m._prior_sqrt(np.eye(m._sqrt_size()))
    nodes, weights, cells = m._interpolation(x)
    on_nodes = np.all(np.isclose(np.sort(weights[:20], axis=1)[:, -1], 1.0))
    recon = np.einsum("pi,pij->pj", weights, v[nodes])
    recon /= np.linalg.norm(recon, axis=1)[:, None]
    print("   Q vs tau^2 L (C^-1 L)^(alpha-1): max relative difference %.1e; F F' vs Q: %.1e; "
          "weights of the points on vertices are 1: %s; max angle between a point and its interpolated position %.1e rad"
          % (np.abs(Q - Qref).max() / np.abs(Qref).max(), np.abs(F @ F.T - Q).max() / np.abs(Q).max(), on_nodes,
             np.arccos(np.clip(np.sum(recon * x, axis=1), -1, 1)).max()))
    return m


def check_normalization(level, nu):
    v, t = _icosphere(level)
    for lengthscale in (0.1, 0.3):
        m = SphereINLAGP(level=level, nu=nu, noise_variance=0.01, amplitude=0.8, lengthscale=lengthscale)
        m.compute(v[:60])
        s2, sigma2, tau2, lam = m._hyperparameters()
        cov = np.linalg.inv(dense_prior(m, m.get_parameter_vector()))
        var = np.diag(cov)
        total, derivative = _sphere_variance_sum(m.alpha, lam[0])
        eps = 1e-5
        fd = (np.log(_sphere_variance_sum(m.alpha, lam[0] * np.exp(eps))[0]) - np.log(_sphere_variance_sum(m.alpha, lam[0] * np.exp(-eps))[0])) / (2 * eps)
        # correlations of a reference node against the Matern of the chordal distance
        i = 12 + np.argmax(np.bincount(t.ravel(), minlength=len(v))[12:] == 6)
        r = np.linalg.norm(v - v[i], axis=1)
        corr = cov[i] / np.sqrt(var[i] * var)
        ref = matern_correlation(r / lengthscale, nu)
        sel = (r > 0.05) & (r < 3 * lengthscale)
        print("check 2 (level %d, nu=%g, l=%g): mean variance / sigma^2 %.4f (min %.4f, max %.4f); d log tau^2 / d log Lambda %.6f vs finite differences %.6f; "
              "correlation vs Matern(chordal): max |difference| %.3f for 0.05 < r < 3 l"
              % (level, nu, lengthscale, var.mean() / sigma2, var.min() / sigma2, var.max() / sigma2, derivative / total, fd, np.abs(corr - ref)[sel].max()))


def check_likelihood(level, nu, rng):
    v, t = _icosphere(level)
    x = np.vstack([v[rng.choice(len(v), 30, replace=False)], random_sphere(rng, 30)])
    y = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + x[:, 2] + 0.05 * rng.standard_normal(len(x)) + 10.0
    m = SphereINLAGP(level=level, nu=nu, noise_variance=0.01, amplitude=0.8, lengthscale=0.25, nsamples=256, seed=1)
    m.compute(x)
    print("check 3-4 (level %d, nu=%g, alpha=%d, %d nodes, %d observations)" % (level, m.nu, m.alpha, m._n, len(y)))
    xtest = np.vstack([v[rng.choice(len(v), 20, replace=False)], random_sphere(rng, 20)])
    for trial in range(2):
        p = params_vector(0.01 * (trial + 1), 0.8 / (trial + 1), 0.25 * (1 + 0.4 * trial))
        m.set_parameter_vector(p)
        ll = m.log_likelihood(y)
        s2 = np.exp(p[0])
        A = m._A.toarray()
        Q = dense_prior(m, p)
        Qc = m._assemble(p).toarray()
        r = y - y.mean()
        cov_y = A @ np.linalg.solve(Q, A.T) + s2 * np.eye(len(y))
        ll_dense = -0.5 * (np.linalg.slogdet(cov_y)[1] + r @ np.linalg.solve(cov_y, r) + len(y) * np.log(2 * np.pi))
        posterior = np.linalg.inv(Qc)
        xhat = posterior @ (A.T @ r / s2)
        nodes, weights, _ = m._interpolation(xtest)
        At = np.zeros((len(xtest), m._n))
        for i in range(len(xtest)):
            At[i, nodes[i]] += weights[i]
        mu_dense = At @ xhat + y.mean()
        cov_dense = At @ posterior @ At.T
        var_dense = np.diag(cov_dense)
        mu, var = m.predict(y, xtest, return_var=True)
        mu2, cov = m.predict(y, xtest, return_cov=True)
        err_var = np.abs(var - var_dense) / var_dense
        print("   trial %d: log|Q| factorization %.10e vs dense %.10e; log|Q_c| %.10e vs dense %.10e"
              % (trial, m.logdet_prior(), np.linalg.slogdet(Q)[1], m._logdet_qc, np.linalg.slogdet(Qc)[1]))
        print("            log-likelihood %.10e vs dense %.10e, relative difference %.1e" % (ll, ll_dense, abs(ll - ll_dense) / abs(ll_dense)))
        print("            predictive mean: max difference %.1e (max |mean| %.1e)" % (np.max(np.abs(mu - mu_dense)), np.max(np.abs(mu_dense))))
        print("            predictive variance (%d samples, ring %d): relative error max %.3f, median %.3f; on nodes max %.3f, elsewhere max %.3f"
              % (m.nsamples, m.rb_ring, err_var.max(), np.median(err_var), err_var[:20].max(), err_var[20:].max()))
        print("            predictive covariance (exact solves): max difference %.1e (max variance %.1e), mean max difference %.1e"
              % (np.max(np.abs(cov - cov_dense)), var_dense.max(), np.max(np.abs(mu2 - mu_dense))))
        m.rb_ring = 0
        m._blocks = None
        m._block_structure = None
        _, var0 = m.predict(y, xtest, return_var=True)
        err0 = np.abs(var0 - var_dense) / var_dense
        print("            predictive variance with ring 0: relative error max %.3f, median %.3f" % (err0.max(), np.median(err0)))
        m.rb_ring = 1
        m._blocks = None
        m._block_structure = None


def check_gradient(level, nu, rng):
    v, t = _icosphere(level)
    x = np.vstack([v[rng.choice(len(v), 150, replace=False)], random_sphere(rng, 150)])
    y = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + x[:, 2] + 0.05 * rng.standard_normal(len(x))
    m = SphereINLAGP(level=level, nu=nu, noise_variance=0.01, amplitude=0.8, lengthscale=0.2)
    m.compute(x)
    p = m.get_parameter_vector()
    step = 1e-5
    shifts = np.eye(len(p)) * step
    grad_ref = np.array([(dense_loglik(m, p + e, y) - dense_loglik(m, p - e, y)) / (2 * step) for e in shifts])
    S_inv = np.linalg.inv(dense_covariance(m, p))
    dS = [(dense_covariance(m, p + e) - dense_covariance(m, p - e)) / (2 * step) for e in shifts]
    fisher_ref = np.array([[0.5 * np.trace(S_inv @ Di @ S_inv @ Dj) for Dj in dS] for Di in dS])
    print("check 5 (level %d, nu=%g, %d nodes, %d observations): gradient and Fisher information vs finite differences of the dense likelihood"
          % (level, nu, m._n, len(y)))
    print("   reference gradient %s" % np.array2string(grad_ref, precision=4))
    for nprobe in (10 ** 6, 64):
        m.nprobe = nprobe
        m._prior_probe_cache = None
        m._L_params = None
        m.set_parameter_vector(p)
        count0 = inla._factorization_count[0]
        start = time.time()
        ll = m.log_likelihood(y)
        grad = m.grad_log_likelihood(y)
        t_grad = time.time() - start
        count1 = inla._factorization_count[0]
        start = time.time()
        fisher = m.fisher_information()
        t_fisher = time.time() - start
        count2 = inla._factorization_count[0]
        label = "exact traces" if nprobe >= len(y) else "%d probes" % nprobe
        print("   %-12s gradient %s, relative error per component %s (%.2f s, %d factorizations for likelihood + gradient)"
              % (label, np.array2string(grad, precision=4), np.array2string(np.abs(grad - grad_ref) / np.abs(grad_ref), formatter={'float_kind': lambda v: '%.1e' % v}), t_grad, count1 - count0))
        print("   %-12s Fisher information: relative error (Frobenius) %.1e (%.2f s, %d factorizations)"
              % (label, np.linalg.norm(fisher - fisher_ref) / np.linalg.norm(fisher_ref), t_fisher, count2 - count1))
        # the likelihood after the Fisher information (Q_c refactored on demand) is unchanged
        print("   %-12s log-likelihood before %.10e, after the Fisher information %.10e" % (label, ll, m.log_likelihood(y)))


if __name__ == "__main__":
    args = parse_args()
    if args.superlu:
        from pdbridge import superlu_factor, superlu_logdet, superlu_solve
        inla._superlu = lambda: (superlu_factor, superlu_logdet, superlu_solve)
    else:
        inla._superlu = lambda: (ScipyBridge.factor, ScipyBridge.logdet, ScipyBridge.solve)
    rng = np.random.default_rng(2026)
    for level, nu in ((3, 1.0), (3, 2.0), (4, 1.0), (4, 3.0)):
        check_mesh(level, nu, rng)
    for level, nu in ((4, 1.0), (4, 2.0)):
        check_normalization(level, nu)
    for level, nu in ((3, 1.0), (3, 2.0), (4, 1.0), (4, 3.0)):
        check_likelihood(level, nu, rng)
    for level, nu in ((3, 1.0), (4, 2.0), (3, 3.0)):
        check_gradient(level, nu, rng)
