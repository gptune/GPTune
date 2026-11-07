#! /usr/bin/env python
"""
Replay the SuperLU work of one sparse-GP likelihood evaluation (factor + logdet + solve through
pdbridge) on a matrix shaped like the N=100k, 2D Schwefel runs of model_comparison_updated_superlu.py:
uniform points in [0,1]^2, Wendland C2 covariance with support equal to the cutoff (0.0158), which
gives about 78 nonzeros per row. The SuperLU workers must already be running (see
bench_superlu_scaling.sbatch). Prints the driver-side time of each pdbridge call.
"""
import sys
import time
import numpy as np
import scipy.sparse as sps
from scipy.spatial import cKDTree
from pdbridge import superlu_factor, superlu_logdet, superlu_solve, superlu_terminate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
CUTOFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0158
NREP = 3

rng = np.random.default_rng(0)
x = rng.random((N, 2))
pairs = cKDTree(x).query_pairs(CUTOFF, output_type='ndarray')
r = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1) / CUTOFF
v = (1 - r)**4 * (4 * r + 1)  # Wendland C2
rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(N)])
cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(N)])
vals = np.concatenate([v, v, np.full(N, 1.0 + 1e-6)])
K = sps.csc_matrix((vals, (rows, cols)), shape=(N, N))
print("N=%d cutoff=%g nnz=%d (%.1f per row)" % (N, CUTOFF, K.nnz, K.nnz / N), flush=True)

b = rng.random(N)
for rep in range(NREP):
    t0 = time.time()
    superlu_factor(K, 0, 0, True)  # INT64=0, algo3d=0 as in george's sparse solver; verbose prints init and factor times
    t1 = time.time()
    sign, logdet = superlu_logdet(True)
    y = b.copy()
    superlu_solve(y, True)
    t2 = time.time()
    res = np.linalg.norm(K @ y - b) / np.linalg.norm(b)
    print("rep %d: factor(init+factor) %.3f s, logdet+solve %.3f s, logdet %.6e, relres %.1e"
          % (rep, t1 - t0, t2 - t1, sign * logdet, res), flush=True)
superlu_terminate()
