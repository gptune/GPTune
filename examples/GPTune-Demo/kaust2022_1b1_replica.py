#! /usr/bin/env python
"""
A replica of dataset 1b-1 of the 2022 KAUST Competition on Spatial Statistics for Large Datasets, from its
generating model in Abdulah et al., "The Second Competition on Spatial Statistics for Large Datasets"
(arXiv:2211.03119, equation (4)): Z(s) = m(s) + eps(s) on a regular grid of 1,000,000 points of [0, 1]^2, with
eps ~ N(0, tau^2), tau = 0.3, and, with u = (s_x + s_y) / 2,

    m(s) = 3 sin{20 (u + 1.9)} cos{20 (u - 1.2)^6} + 3/5 exp{sin(25 s_x) + sin(13 s_y)} + (u - 0.2) / 2

randomly split into 90% training and 10% testing values (the original data are on Kaggle). Only the noise
realization, the split and the exact grid nodes (here i / 999) differ from the original data.

Usage: python kaust2022_1b1_replica.py [output.npz]   (default $PSCRATCH/kaust2022/kaust2022_1b1_replica.npz)
Writes xtrain, ytrain, xtest, ytest (noisy values, as scored by the competition) and mtest, the noise-free mean
at the testing locations.
"""

import os
import sys

import numpy as np

TAU = 0.3
NGRID = 1000


def mean_function(s):
    u = (s[:, 0] + s[:, 1]) / 2
    return (3 * np.sin(20 * (u + 1.9)) * np.cos(20 * (u - 1.2) ** 6)
            + 0.6 * np.exp(np.sin(25 * s[:, 0]) + np.sin(13 * s[:, 1])) + 0.5 * (u - 0.2))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ['PSCRATCH'], 'kaust2022/kaust2022_1b1_replica.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rng = np.random.default_rng(2022)
    g = np.arange(NGRID) / (NGRID - 1)
    s = np.stack([np.repeat(g, NGRID), np.tile(g, NGRID)], axis=1)
    m = mean_function(s)
    z = m + TAU * rng.standard_normal(len(m))
    test = np.zeros(len(m), dtype=bool)
    test[rng.choice(len(m), len(m) // 10, replace=False)] = True
    np.savez(out, xtrain=s[~test], ytrain=z[~test], xtest=s[test], ytest=z[test], mtest=m[test])
    print("%s: %d training and %d testing values; m(s) in [%.3f, %.3f], variance %.3f; tau %.1f"
          % (out, (~test).sum(), test.sum(), m.min(), m.max(), m.var(), TAU))


if __name__ == "__main__":
    main()
