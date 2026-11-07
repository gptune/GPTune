#! /usr/bin/env python
"""
Converts a dataset of sub-competition 2b of the 2023 KAUST Competition on Spatial Statistics for Large Datasets
(2023-kaust-competition-spatial-statistics-large-datasets.zip, unpacked) to the npz read by kaust_inla_wendland.py:
xtrain, ytrain (900,000 values), xtest, ytest (testing set 1, distributed as the training locations) and, for
datasets 1-4, xtest2, ytest2 (testing set 2, uniform locations), 100,000 values each. The testing sets are randomly
permuted, so that their first values (where the driver computes the predictive variances) are a random subset.

True models (stationary isotropic Matern with a nugget, "Competition Description and True Model.pdf", Table 2):
  1 Chessboard   sigma^2 1.4  beta 0.164056  nu 1.6  tau^2 0.27
  2 Left-bottom  sigma^2 1.4  beta 0.052579  nu 2.3  tau^2 0.27
  3 Satellite    sigma^2 1.7  beta 0.021707  nu 1.4  tau^2 0.27
  4 Clusters     sigma^2 1.5  beta 0.054710  nu 2.1  tau^2 0.27
  5 Regular      sigma^2 1.7  beta 0.021707  nu 1.4  tau^2 0.27

Usage: python kaust2023_2b_npz.py <dataset 1-5> [unpacked directory] [output.npz]
  (defaults $PSCRATCH/kaust2023/2023-kaust-competition-spatial-statistics-large-datasets and
   $PSCRATCH/kaust2023/kaust2023_2b_dataset0<d>.npz)
"""

import os
import sys

import numpy as np


def main():
    d = int(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.environ['PSCRATCH'], 'kaust2023/2023-kaust-competition-spatial-statistics-large-datasets')
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.environ['PSCRATCH'], 'kaust2023/kaust2023_2b_dataset%02d.npz' % d)
    sub = os.path.join(base, 'Sub-competition_2b')
    read = lambda name: np.loadtxt(os.path.join(sub, name), delimiter=',', ndmin=1)
    arrays = {'xtrain': read('Training data/M_LOC_Train_%d.csv' % d), 'ytrain': read('Training data/M_Z_Train_%d.csv' % d)}
    rng = np.random.default_rng(2023)
    for t, suffix in ((1, ''), (2, '2')):
        name = 'Testing set %d/M_LOC_Test%d_%d.csv' % (t, t, d)
        if not os.path.exists(os.path.join(sub, name)):
            continue
        x, y = read(name), read('Testing set %d/M_Z_Test%d_%d.csv' % (t, t, d))
        p = rng.permutation(len(y))
        arrays['xtest' + suffix], arrays['ytest' + suffix] = x[p], y[p]
    np.savez(out, **arrays)
    x = arrays['xtrain']
    print("%s: %s; training locations in [%.3f, %.3f] x [%.3f, %.3f], %.1f%% in [0, 0.5]^2"
          % (out, ', '.join('%s %d' % (k, len(v)) for k, v in arrays.items() if k.startswith('y')),
             x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max(), 100 * np.mean((x < 0.5).all(axis=1))))


if __name__ == "__main__":
    main()
