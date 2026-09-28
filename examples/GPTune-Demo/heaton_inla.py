#! /usr/bin/env python
"""
The MODIS land surface temperature case study of Heaton et al. (2019), "A Case Study Competition
Among Methods for Analyzing Large Spatial Data", with the INLA/SPDE model of Model_George
(model_kern='INLA', george/inla.py) and SuperLU_DIST.

The data (https://github.com/finnlindgren/heatoncomparison, Data/All*Temps.RData) are on a 500 x 300
lon/lat lattice, which is the lattice of the model. The training data are the non-missing MaskTemp
values (105,569), the test data the TrueTemp values where MaskTemp is missing (42,740 for the
satellite data, 44,431 for the simulated data). The coordinates are normalized to [0, 1] per
dimension as in GPTune, with one length scale per dimension. The hyperparameters are estimated by
L-BFGS with finite differences, and the test predictions are scored as in the paper: MAE, RMSE, CRPS,
interval score and coverage of the 95% prediction intervals (of the observations, i.e. with the
noise variance added to the predictive variance of the latent field).

Reading the RData files needs pyreadr (pip install pyreadr). The SuperLU_DIST workers are launched
separately, see heaton_inla.sbatch.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

from pdbridge import superlu_factor, superlu_logdet, superlu_solve, superlu_terminate
# the INLA model calls superlu_factor/logdet/solve through george.solvers.basic (see model_comparison_updated_superlu.py)
import george.solvers.basic
for _f in (superlu_factor, superlu_logdet, superlu_solve):
    setattr(george.solvers.basic, _f.__name__, _f)

from GPTune.options import Options
from GPTune.model import Model_George
from GPTune.data import Data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, required=True, help='The Data directory of the heatoncomparison repository')
    parser.add_argument('-dataset', type=str, default='satellite', choices=['satellite', 'simulated'])
    parser.add_argument('-window', type=int, nargs=2, default=None,
                        help='Use only the first NLON x NLAT nodes of the lattice (for quick tests)')
    parser.add_argument('-nsamples', type=int, default=128, help='Posterior samples of the variance estimator (model_inla_nsamples)')
    parser.add_argument('-buffer', type=float, default=None, help='model_inla_buffer (default: twice the largest length scale)')
    parser.add_argument('-lengthscale_max', type=float, default=1.0, help='Largest length scale (normalized coordinates)')
    parser.add_argument('-out', type=str, default='heaton_inla', help='Prefix of the output files')
    parser.add_argument('-terminate', type=int, default=1, help='Whether to terminate the SuperLU_DIST workers at the end')
    return parser.parse_args()


def load(data_dir, dataset, window):
    import pyreadr
    name = {'satellite': 'AllSatelliteTemps', 'simulated': 'AllSimulatedTemps'}[dataset]
    frame = next(iter(pyreadr.read_r(os.path.join(data_dir, name + '.RData')).values()))
    lon, lat = frame['Lon'].to_numpy(float), frame['Lat'].to_numpy(float)
    train_temp, true_temp = frame['MaskTemp'].to_numpy(float), frame['TrueTemp'].to_numpy(float)
    lons, lats = np.unique(lon), np.unique(lat)
    if window is not None:
        keep = (lon <= lons[window[0] - 1]) & (lat <= lats[window[1] - 1])
        lon, lat, train_temp, true_temp = lon[keep], lat[keep], train_temp[keep], true_temp[keep]
        lons, lats = lons[:window[0]], lats[:window[1]]
    # normalized coordinates: the lattice nodes are i / (nlon - 1) and j / (nlat - 1)
    x = np.stack([(lon - lons[0]) / (lons[-1] - lons[0]), (lat - lats[0]) / (lats[-1] - lats[0])], axis=1)
    train = ~np.isnan(train_temp)
    test = np.isnan(train_temp) & ~np.isnan(true_temp)
    degrees = np.array([lons[-1] - lons[0], lats[-1] - lats[0]])
    return x[train], train_temp[train], x[test], true_temp[test], (len(lons), len(lats)), degrees


def scores(y, mu, var):
    """The scores of Heaton et al.: MAE, RMSE, CRPS, 95% interval score and coverage."""
    sd = np.sqrt(var)
    z = (y - mu) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    alpha = 0.05
    low, high = mu - norm.ppf(1 - alpha / 2) * sd, mu + norm.ppf(1 - alpha / 2) * sd
    interval = (high - low) + 2 / alpha * (low - y) * (y < low) + 2 / alpha * (y - high) * (y > high)
    return {'MAE': np.mean(np.abs(y - mu)), 'RMSE': np.sqrt(np.mean((y - mu) ** 2)), 'CRPS': np.mean(crps),
            'INT': np.mean(interval), 'CVG': np.mean((y >= low) & (y <= high))}


def main():
    args = parse_args()
    xtrain, ytrain, xtest, ytest, shape, degrees = load(args.data_dir, args.dataset, args.window)
    print("Heaton %s data: lattice %d x %d, %d training and %d test values" % (args.dataset, shape[0], shape[1], len(ytrain), len(ytest)))

    options = Options()
    options.update(model_class='Model_George', model_kern='INLA', model_inla_shape=list(shape),
                   model_inla_buffer=args.buffer, model_inla_nsamples=args.nsamples,
                   model_noisevariance=[1e-4, 10.0, 0.1], model_amplitude=[1e-2, 1e3, 10.0],
                   model_lengthscale=[1e-3, args.lengthscale_max, 0.05],
                   model_mcmc=False, model_grad=False, model_random_seed=0, verbose=False,
                   model_history_file=args.out + '_training.csv')
    data = Data(problem=None)
    data.I, data.P, data.O = [[0]], [xtrain], [ytrain[:, None]]
    model = Model_George(problem=None, computer=None)
    start = time.time()
    hyperparameters, modeling_options, model_stats, nfev = model.train(data, **options)
    train_time = time.time() - start
    lengthscale_degrees = np.array(hyperparameters['lengthscale']) * degrees
    print("training: %.1f s, %d likelihood evaluations, lattice with buffer %s; noise variance %.4g, variance %.4g, "
          "length scales %s (degrees %s), log-likelihood %.8e"
          % (train_time, nfev, "x".join(map(str, model.M._shape)), hyperparameters['noise_variance'][0],
             hyperparameters['variance'][0], np.round(hyperparameters['lengthscale'], 5), np.round(lengthscale_degrees, 4),
             model_stats['log_marginal_likelihood']))

    start = time.time()
    mu, var = model.predict(xtest, tid=0)
    predict_time = time.time() - start
    mu, var = mu[:, 0], var[:, 0] + hyperparameters['noise_variance'][0]
    result = scores(ytest, mu, var)
    print("prediction of %d test values: %.1f s (including the %d posterior samples)" % (len(ytest), predict_time, args.nsamples))
    print("scores: " + ", ".join("%s %.4f" % item for item in result.items()))

    np.savez(args.out + '_predictions.npz', xtest=xtest, ytest=ytest, mu=mu, var=var)
    with open(args.out + '_stats.json', 'w') as f:
        json.dump({'dataset': args.dataset, 'lattice': shape, 'ntrain': len(ytrain), 'ntest': len(ytest),
                   'train_time': train_time, 'predict_time': predict_time, 'nfev': nfev,
                   'hyperparameters': hyperparameters, 'lengthscale_degrees': lengthscale_degrees,
                   'log_marginal_likelihood': model_stats['log_marginal_likelihood'], 'scores': result,
                   'modeling_options': modeling_options}, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    if args.terminate:
        superlu_terminate()


if __name__ == "__main__":
    main()
