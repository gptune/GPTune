#!/usr/bin/env python
"""
Standalone reproducer of a wrong determinant sign from ButterflyPACK's H2 factorization (format 7, GPU),
using only numpy and ButterflyPACK's Python interface (dPy_BPACK_wrapper.py driver side,
dPy_BPACK_worker.py workers; run_logdet_sign_repro.sbatch starts both).

Matrix, for n of the 900K 2D points in points.npy (the KAUST 2023 dataset-2 locations: 94% of them in the
bottom-left quarter of [0,1]^2), chosen as np.random.default_rng(0).choice(900000, n, replace=False):

    K(i, j) = exp(-||x_i - x_j|| / 0.01) + 0.05 [i == j]

The exponential (Matern-1/2) kernel is positive definite in any dimension, so K >= 0.05 I and det K > 0:
the sign must be +1. ButterflyPACK reports -1 at some sizes (100K and 400K with the defaults of the sbatch:
16 ranks, nmin_leaf 8, unstructured tree), both at factor time (worker log line "logdet: -1 ...") and from
c_bpack_logdet (bpack_logdet below), with the right magnitude. The solve is fine.

For each size: factor, bpack_logdet, the residual of one solve (K applied with bpack_mult), and up to
--dense-max points the exact log det K from a dense Cholesky factorization.

Usage (inside run_logdet_sign_repro.sbatch): python logdet_sign_repro.py --sizes 20000,100000
"""
import argparse
import os
import sys
import time

import numpy as np
from dPy_BPACK_wrapper import bpack_factor, bpack_free, bpack_logdet, bpack_mult, bpack_solve, bpack_terminate

HERE = os.path.dirname(os.path.abspath(__file__))

# The GPU entry evaluator (compiled by ButterflyPACK with NVRTC): george's exponential kernel, mode 0.
# params: amplitude, mode (0), the inverse squared length scale per dimension, then the diagonal term per
# point (by 0-based global id).
ENTRY_SOURCE = r"""
__device__ double bpack_entry(const double* x, long long i, const double* y, long long j,
                              const double* params, int dim) {
    double q = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double t = x[d] - y[d];
        q += params[2 + d] * t * t;
    }
    double v = params[0] * exp(-sqrt(q));
    if (i == j) v += params[2 + dim + i];
    return v;
}
"""
BPACK_GPU_SYMMETRIC = 1
BPACK_GPU_COORDINATES = 2


def dense_logdet(x, amplitude, ell, noise, chunk=2000):
    """log det K from a dense Cholesky factorization (n^2 doubles)."""
    n = len(x)
    K = np.empty((n, n))
    for s in range(0, n, chunk):
        d = x[s:s + chunk, None, :] - x[None, :, :]
        K[s:s + chunk] = amplitude * np.exp(-np.sqrt(np.einsum("ijk,ijk->ij", d, d)) / ell)
    K[np.diag_indices(n)] += noise
    L = np.linalg.cholesky(K)  # (raises if K were not positive definite)
    return 2.0 * np.sum(np.log(np.diag(L)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="20000,100000", help="numbers of points, comma separated")
    parser.add_argument("--points", default=os.path.join(HERE, "points.npy"))
    parser.add_argument("--ell", type=float, default=0.01)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--dense-max", type=int, default=20000, help="largest size with the dense reference")
    args = parser.parse_args()

    xall = np.load(args.points)
    dim = xall.shape[1]
    print("points %s: %d x %d; K(i,j) = %g exp(-|x_i - x_j| / %g) + %g [i == j]"
          % (args.points, xall.shape[0], dim, args.amplitude, args.ell, args.noise), flush=True)
    wrong = 0
    try:
        for n in [int(v) for v in args.sizes.split(",")]:
            keep = np.random.default_rng(0).choice(len(xall), n, replace=False)
            x = np.ascontiguousarray(xall[keep], dtype=np.float64)
            yerr2 = np.full(n, args.noise)
            meta = {"coordinates": x, "amplitude": args.amplitude, "ell": args.ell, "yerr2": yerr2}
            params = np.concatenate(([args.amplitude, 0.0], np.full(dim, 1.0 / args.ell ** 2), yerr2))
            payload = {
                "block_func_filepath": os.path.join(HERE, "repro_block_funcs.py"),
                "block_func_module": "repro_block_funcs",
                "block_func_name": "compute_block",
                "meta": meta,
                "gpu_entry": {"source": ENTRY_SOURCE, "params": params,
                              "flags": BPACK_GPU_SYMMETRIC | BPACK_GPU_COORDINATES},
            }
            t0 = time.time()
            bpack_factor(payload, fid=0)
            sign, logdet = bpack_logdet(fid=0)
            b = np.random.default_rng(1).standard_normal(n)
            a = np.asarray(bpack_solve(b, fid=0)).ravel()
            residual = np.linalg.norm(np.asarray(bpack_mult(a, "N", fid=0)).ravel() - b) / np.linalg.norm(b)
            seconds = time.time() - t0
            dense = ""
            if n <= args.dense_max:
                exact = dense_logdet(x, args.amplitude, args.ell, args.noise)
                dense = " | dense log det K %.10e (H2 log|det| - dense %+.2e)" % (exact, logdet - exact)
            status = "WRONG SIGN (K is SPD)" if sign < 0 else "ok"
            wrong += sign < 0
            print("RESULT n %d: bpack_logdet sign %+.0f, log|det| %.10e | solve residual %.1e%s | %.1f s | %s"
                  % (n, sign, logdet, residual, dense, seconds, status), flush=True)
            bpack_free(fid=0)
    finally:
        bpack_terminate()
    print("done: %d size(s) with a wrong sign" % wrong, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
