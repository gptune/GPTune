#!/usr/bin/env python
"""
Gradient check of george's H2 (butterflypack, model_bpack=1) Gaussian-process likelihood, evaluated as
Model_George evaluates it (GP.log_likelihood and GP.grad_log_likelihood with BasicSolver, white noise fitted,
yerr = model_jitter = 1e-10). For each subsample size of a dataset, at the hyperparameters x0:
  - the analytic gradient (Hutchinson traces with fixed Rademacher probes; GC_NPROBES probe counts, the first
    one is Model_George's 64),
  - central differences of the H2 log-likelihood in each parameter, with steps h and h/4 (their agreement shows
    whether the H2 likelihood is smooth at that scale),
  - up to GC_DENSE_MAX points, george's exact dense likelihood and exact-trace gradient (BasicSolver with
    model_bpack=0: Cholesky, K^-1 formed).
Run inside run_spatial_h2_gpu.sbatch (DRIVER=gradcheck), which starts the butterflypack workers; the worker log
has the H2_CheckError accuracies (acc_mvp of the compression, acc_forward of the factorization's solve) of every
factorization. Lines starting with "GC " are the results; GC_OUT (default gradient_check.json in the results
directory) collects them after each size.

Environment:
  GC_DATA          npz with xtrain, ytrain (default the rough Matern-1/2 field on the KAUST 2023 dataset-2 design)
  GC_SIZES         subsample sizes, comma separated (default 10000,20000,50000,100000,200000,400000)
  GC_KERN          RBF, Exp, Matern32 or Matern52 (default Exp)
  GC_NOISE         noise variance at x0 (default 0.05)
  GC_AMPLITUDE     kernel variance at x0 (default 1.0)
  GC_LENGTHSCALE   length scale(s) at x0, one value or one per dimension (default 0.01)
  GC_H             central-difference step in the log parameters (default 1e-3; also h/4)
  GC_NPROBES       Hutchinson probe counts, comma separated (default 64,256)
  GC_DENSE_MAX     largest size with the dense reference (default 20000)
  GC_FD            1 (default) for the central differences, 0 to skip them
At x0 the solve is also checked: r^T K^-1 r and the residual ||K a - r|| / ||r|| (K applied with the
compressed operator) for a from a single-column solve (as GP.log_likelihood's dot_solve) and from the
multi-column solve [r, U] of GP.grad_log_likelihood (U: the 64 fixed Rademacher probes).
"""
import os
import sys
import json
import time
import numpy as np

GPTUNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, GPTUNE_ROOT)
import george
george.__path__.insert(0, os.path.join(GPTUNE_ROOT, 'george/src/george'))
from george.solvers import BasicSolver

KERNELS = {'RBF': george.kernels.ExpSquaredKernel, 'Exp': george.kernels.ExpKernel,
           'Matern32': george.kernels.Matern32Kernel, 'Matern52': george.kernels.Matern52Kernel}
JITTER = 1e-10  # Model_George's model_jitter, passed as yerr


def make_gp(x, kern, noise, amplitude, lengthscale, bpack, nprobe):
    """A george GP as Model_George builds it (anisotropic metric, kernel times amplitude, white noise fitted)."""
    ndim = x.shape[1]
    kernel = KERNELS[kern](metric=np.asarray(lengthscale, dtype=float) ** 2, ndim=ndim)
    kernel *= amplitude
    solver_kwargs = {'verbose': 0, 'compute_grad': 1, 'nprobe': int(nprobe), 'model_sparse': 0,
                     'model_bpack': int(bpack), 'bpack_scaled_geometry': 0, 'debug': 0, 'sym': 0}
    gp = george.GP(kernel=kernel, white_noise=np.log(noise), fit_white_noise=True, solver=BasicSolver,
                   **solver_kwargs)
    gp.compute(x, None, yerr=JITTER)
    return gp


def nll(gp, y, p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y, quiet=False)


def main():
    data = os.environ.get('GC_DATA', os.path.join(os.environ.get('PSCRATCH', '.'), 'kaust2023/rough_ds2_nu0.5_r0.02.npz'))
    sizes = [int(v) for v in os.environ.get('GC_SIZES', '10000,20000,50000,100000,200000,400000').split(',')]
    kern = os.environ.get('GC_KERN', 'Exp')
    noise = float(os.environ.get('GC_NOISE', 0.05))
    amplitude = float(os.environ.get('GC_AMPLITUDE', 1.0))
    h = float(os.environ.get('GC_H', 1e-3))
    nprobes = [int(v) for v in os.environ.get('GC_NPROBES', '64,256').split(',')]
    dense_max = int(os.environ.get('GC_DENSE_MAX', 20000))
    out = os.environ.get('GC_OUT', os.path.join(os.environ.get('GP_OUTPUT_DIR', '.'), 'gradient_check.json'))

    z = np.load(data)
    xall, yall = z['xtrain'], z['ytrain']
    ndim = xall.shape[1]
    ls = [float(v) for v in os.environ.get('GC_LENGTHSCALE', '0.01').split()]
    lengthscale = ls * ndim if len(ls) == 1 else ls
    print("GC data %s (%d points, %d dims), kernel %s, x0: noise %g amplitude %g length scales %s, h %g, probes %s"
          % (data, len(yall), ndim, kern, noise, amplitude, lengthscale, h, nprobes), flush=True)
    results = []
    for n in sizes:
        if n > len(yall):
            continue
        keep = np.random.default_rng(0).choice(len(yall), n, replace=False)
        x, y = xall[keep], yall[keep] - np.mean(yall[keep])
        row = {'n': n}
        t0 = time.time()
        gp = make_gp(x, kern, noise, amplitude, lengthscale, True, nprobes[0])
        p0 = gp.get_parameter_vector().copy()
        names = gp.get_parameter_names()
        row['names'], row['x0'] = list(names), p0.tolist()
        row['nll'] = nll(gp, y, p0)
        # the solve of the likelihood (one column) against the solve of the gradient ([r, U], 65 columns)
        gp.set_parameter_vector(p0)
        gp.recompute()
        r = np.ascontiguousarray(y, dtype=np.float64)
        a1 = np.asarray(gp.solver.apply_inverse(r.copy(), in_place=False)).ravel()
        U = np.random.default_rng(0).choice([-1.0, 1.0], size=(n, 64))
        aM = np.asarray(gp.solver.apply_inverse(np.column_stack([r, U]), in_place=False)).reshape(n, -1)[:, 0]
        solve = {}
        for tag, a in (('single', a1), ('multi', aM)):
            res = np.linalg.norm(np.asarray(gp.solver.apply_forward(a, 0)).ravel() - r) / np.linalg.norm(r)
            solve[tag] = {'rKr': float(r @ a), 'residual': float(res)}
        solve['alpha_rel_diff'] = float(np.linalg.norm(a1 - aM) / np.linalg.norm(aM))
        row['solve'] = solve
        print("GC n %d solve: r^T K^-1 r single %.10e (residual %.2e) | multi %.10e (residual %.2e) | ||a1-aM||/||aM|| %.2e"
              % (n, solve['single']['rKr'], solve['single']['residual'], solve['multi']['rKr'], solve['multi']['residual'],
                 solve['alpha_rel_diff']), flush=True)
        grads = {}
        for k in nprobes:
            gp.solver_kwargs['nprobe'] = k  # (the solver is rebuilt from solver_kwargs at each recompute)
            gp.solver.nprobe = k
            gp.set_parameter_vector(p0)
            grads[k] = (-gp.grad_log_likelihood(y, quiet=False)).tolist()
        gp.solver_kwargs['nprobe'] = nprobes[0]
        gp.solver.nprobe = nprobes[0]
        row['grad'] = grads
        fd = {}
        for step in ((h, h / 4) if os.environ.get('GC_FD', '1') == '1' else ()):
            g = []
            for i in range(len(p0)):
                pp, pm = p0.copy(), p0.copy()
                pp[i] += step
                pm[i] -= step
                g.append((nll(gp, y, pp) - nll(gp, y, pm)) / (2 * step))
            fd[step] = g
        row['fd'] = {('%g' % k): v for k, v in fd.items()}
        row['h2_seconds'] = time.time() - t0
        print("GC n %d: H2 -loglik %.10e (%.1f s)" % (n, row['nll'], row['h2_seconds']), flush=True)
        for i, name in enumerate(names):
            central = ' | central h %+.6e  h/4 %+.6e' % (fd[h][i], fd[h / 4][i]) if fd else ''
            print("GC n %d %-24s analytic %s%s"
                  % (n, name, '  '.join('p%d %+.6e' % (k, grads[k][i]) for k in nprobes), central), flush=True)
        if n <= dense_max:
            t0 = time.time()
            gd = make_gp(x, kern, noise, amplitude, lengthscale, False, nprobes[0])
            gd.set_parameter_vector(p0)
            row['dense_nll'] = -gd.log_likelihood(y, quiet=False)
            row['dense_grad'] = (-gd.grad_log_likelihood(y, quiet=False)).tolist()
            row['dense_seconds'] = time.time() - t0
            print("GC n %d: dense -loglik %.10e (H2 - dense %+.3e; %.1f s)"
                  % (n, row['dense_nll'], row['nll'] - row['dense_nll'], row['dense_seconds']), flush=True)
            for i, name in enumerate(names):
                print("GC n %d %-24s dense exact %+.6e" % (n, name, row['dense_grad'][i]), flush=True)
        results.append(row)
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
    print("GC done", flush=True)


if __name__ == '__main__':
    main()
