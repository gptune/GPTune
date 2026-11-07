#! /usr/bin/env python
"""
The MODIS land surface temperature case study of Heaton et al. (2019), "A Case Study Competition Among
Methods for Analyzing Large Spatial Data", with the two sparse GP models of george with SuperLU_DIST, trained
by GPTune's Model_George with up to four optimizers (the settings of model_comparison_updated_superlu.py):
  wendland  the Wendland C2 kernel (sparse covariance matrix), with its cutoff of about 78 neighbours per sample
  inla      the INLA/SPDE Matern model (sparse precision matrix, george/inla.py) on the 500 x 300 lattice of the data
and with the dense RBF kernel compressed by butterflypack:
  h2        the H2 format on GPUs (run_spatial_h2_gpu.sbatch runs this driver with the butterflypack workers)

The data are in the Data directory of https://github.com/finnlindgren/heatoncomparison (All*Temps.RData, read
with pyreadr, or their conversions heaton_satellite.npz / heaton_simulated.npz with the arrays lon, lat,
train_temp = MaskTemp and true_temp = TrueTemp). The training data are the non-missing MaskTemp values
(105,569), the test data the TrueTemp values where MaskTemp is missing (42,740 for the satellite data, 44,431
for the simulated data). As in GPTune, the coordinates are normalized to [0, 1] per dimension, with one length
scale per dimension; both models are trained on the temperatures minus their training mean. The test
predictions are scored as in the paper: MAE, RMSE, CRPS, and interval score and coverage of the 95% prediction
intervals (of the observations: the noise variance is added to the predictive variance).

The environment selects the runs (run_heaton_gpu.sbatch runs this driver with the SuperLU_DIST workers):
  HEATON_DATA     the data directory (default $PSCRATCH/heatoncomparison/Data)
  HEATON_DATASET  satellite (default) or simulated
  GP_MODELS       default "wendland,inla"
  GP_OPTIMIZERS   default "gradient,finite difference,mcmc,mala" (the L-BFGS trainings first: they set the
                  time limit of the samplers, 3 times the shorter one of the model)
  HEATON_WINDOW   "NLON NLAT": only the first NLON x NLAT nodes of the lattice (for quick tests)
  HEATON_INLA_NU  the smoothness nu of the INLA Matern field (default 1; 0.5 is the exponential covariance
                  of the SPDE entry of the paper, approximated as by R-INLA)
  HEATON_WENDLAND_CUTOFF  the largest Wendland support radius (normalized units), which sets the sparsity of
                  the covariance matrix (default: about 78 neighbours per sample, 0.0154 for the full data; the
                  test cells of the satellite data are within 0.0154 of a training cell for 68% of them, within
                  0.05 for 94%)
  GP_OUTPUT_DIR   the directory of the results (default .)
Writes heaton_<dataset>_compare.csv/json and, per model, the training histories and the test predictions of
every optimizer in GP_OUTPUT_DIR/<model>.
"""

import sys
import os
# GPTune and george (for its INLA model, george/inla.py) of the source trees of this repository, which the
# installed packages may predate
GPTUNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, GPTUNE_ROOT)
import george
george.__path__.insert(0, os.path.join(GPTUNE_ROOT, 'george/src/george'))
import json
import time

import numpy as np
from scipy.stats import norm

from pdbridge import superlu_factor, superlu_logdet, superlu_solve
# the sparse george models call superlu_factor/logdet/solve through george.solvers.basic
import george.solvers.basic
for _f in (superlu_factor, superlu_logdet, superlu_solve):
    setattr(george.solvers.basic, _f.__name__, _f)

from GPTune.options import Options
from GPTune.model import Model_George
from GPTune.data import Data

PREDICT_CHUNK = 1000 # test points predicted together (the Wendland cross-covariance of a chunk is dense)
H2_PREDICT_CHUNK = 250 # for h2: the right-hand sides of one butterflypack solve (as model_comparison_updated_bpack.py)


def load(data_dir, dataset, window):
    """Normalized training and test coordinates and temperatures, the lattice shape, and its size in degrees."""
    npz = os.path.join(data_dir, 'heaton_%s.npz' % dataset)
    if os.path.exists(npz):
        z = np.load(npz)
        lon, lat, train_temp, true_temp = z['lon'], z['lat'], z['train_temp'], z['true_temp']
    else:
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


def wendland_cutoff(ntrain):
    """The largest Wendland support radius: HEATON_WENDLAND_CUTOFF, or about 78 neighbours per sample."""
    return float(os.environ.get('HEATON_WENDLAND_CUTOFF', 0.0158 * np.sqrt(100000 / ntrain)))


def model_options(model_name, optimizer, ntrain, shape, history_file, lbfgs_times):
    """The Model_George options of a model and an optimizer."""
    options = Options()
    options.update(model_class='Model_George', model_noisevariance=[1e-4, 10.0, 0.1], model_amplitude=[1e-2, 1e3, 10.0],
                   model_lengthscale=[1e-3, 1.0, 0.05], model_random_seed=0, verbose=True, debug=False,
                   model_history_file=history_file)
    if model_name == 'h2':
        # the dense RBF kernel (the kernel of the GPU entry evaluator of george) in butterflypack's H2 format, with
        # the compression settings of the workers (run_spatial_h2_gpu.sbatch); the smallest length scale is the
        # floor of the INLA model (2 lattice spacings, 1 / 499 along the longitudes): the first L-BFGS step can
        # go to the bound, where the H2 compressions of the derivative operators are slow
        options.update(model_bpack=True, model_kern='RBF', model_isotropic=False, model_lengthscale=[4e-3, 1.0, 0.05])
    elif model_name == 'wendland':
        options.update(model_sparse=True, model_kern='WendlandC2', model_cutoff=wendland_cutoff(ntrain))
    elif model_name == 'inla':
        options.update(model_kern='INLA', model_inla_shape=list(shape),
                       model_inla_nu=float(os.environ['HEATON_INLA_NU']) if os.environ.get('HEATON_INLA_NU') else None)
    else:
        raise ValueError("unknown model %s" % model_name)
    if optimizer == 'gradient':
        options.update(model_mcmc=False, model_grad=True, model_grad_nprobe=64)
    elif optimizer == 'finite difference':
        options.update(model_mcmc=False, model_grad=False)
    elif optimizer == 'mcmc':
        options.update(model_mcmc=True, model_grad=False, model_mcmc_sampler='MetropolisHastings', model_mcmc_nchain=2,
                       model_mcmc_burnin=100, model_mcmc_maxiter=500)
    elif optimizer == 'mala':
        options.update(model_mcmc=True, model_grad=True, model_grad_nprobe=64, model_mcmc_sampler='MALA', model_mcmc_nchain=2,
                       model_mcmc_burnin=20, model_mcmc_maxiter=150, model_mala_fisher_interval=1, model_mala_step_size=1.0)
    else:
        raise ValueError("unknown optimizer %s" % optimizer)
    if optimizer in ('mcmc', 'mala') and len(lbfgs_times) == 2:
        # the samplers stop after 3 times the shorter L-BFGS training of the model
        options.update(model_mcmc_max_time=3 * min(lbfgs_times), model_mcmc_maxiter=1000000)
    return options


def write_summary(output_dir, dataset, summary, info):
    """heaton_<dataset>_compare.json/csv and the table of the runs done so far."""
    prefix = os.path.join(output_dir, 'heaton_%s_compare' % dataset)
    with open(prefix + '.json', 'w') as f:
        json.dump(dict(info, runs=summary), f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    keys = ['train_time', 'likelihood_evaluations', 'time_per_evaluation', 'predict_time', 'MAE', 'RMSE', 'CRPS', 'INT', 'CVG']
    with open(prefix + '.csv', 'w') as f:
        f.write('model,optimizer,' + ','.join(keys) + '\n')
        for row in summary:
            f.write('%s,%s,' % (row['model'], row['optimizer']) + ','.join('%.6g' % row[k] for k in keys) + '\n')
    print("Heaton %s data, %d training and %d test values:" % (dataset, info['ntrain'], info['ntest']))
    print("  %-9s %-18s %10s %6s %9s %9s %8s %8s %8s %8s %6s"
          % ('model', 'optimizer', 'train (s)', 'evals', 'per eval', 'pred (s)', 'MAE', 'RMSE', 'CRPS', 'INT', 'CVG'))
    for row in summary:
        print("  %-9s %-18s %10.1f %6d %9.3f %9.1f %8.4f %8.4f %8.4f %8.4f %6.3f"
              % tuple([row['model'], row['optimizer']] + [row[k] for k in keys]))


def main():
    data_dir = os.environ.get('HEATON_DATA', os.path.join(os.environ.get('PSCRATCH', '.'), 'heatoncomparison/Data'))
    dataset = os.environ.get('HEATON_DATASET', 'satellite')
    models = os.environ.get('GP_MODELS', 'wendland,inla').split(',')
    optimizers = os.environ.get('GP_OPTIMIZERS', 'gradient,finite difference,mcmc,mala').split(',')
    window = [int(v) for v in os.environ['HEATON_WINDOW'].split()] if os.environ.get('HEATON_WINDOW') else None
    output_dir = os.environ.get('GP_OUTPUT_DIR', '.')

    xtrain, ytrain, xtest, ytest, shape, degrees = load(data_dir, dataset, window)
    ymean = np.mean(ytrain)
    print("Heaton %s data: lattice %d x %d, %d training and %d test values, training mean %.4f"
          % (dataset, shape[0], shape[1], len(ytrain), len(ytest), ymean))
    data = Data(problem=None)
    data.I, data.P, data.O = [[0]], [xtrain], [(ytrain - ymean)[:, None]]

    info = {'dataset': dataset, 'lattice': shape, 'degrees': degrees, 'ntrain': len(ytrain), 'ntest': len(ytest),
            'training_mean': ymean, 'window': window, 'wendland_cutoff': wendland_cutoff(len(ytrain)),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    summary = []
    for model_name in models:
        directory = os.path.join(output_dir, model_name)
        os.makedirs(directory, exist_ok=True)
        lbfgs_times = []
        for optimizer in optimizers:
            tag = 'heaton_%s_%s' % (dataset, optimizer.replace(' ', '_'))
            options = model_options(model_name, optimizer, len(ytrain), shape,
                                    os.path.join(directory, 'training_iterations_%s.csv' % tag), lbfgs_times)
            print("=== %s, %s: %s" % (model_name, optimizer, {k: options[k] for k in ('model_kern', 'model_grad', 'model_mcmc', 'model_mcmc_max_time')
                                                             if k in options}))
            model = Model_George(problem=None, computer=None)
            start = time.time()
            hyperparameters, modeling_options, model_stats, nfev = model.train(data, **options)
            train_time = time.time() - start
            if optimizer in ('gradient', 'finite difference'):
                lbfgs_times.append(train_time)

            start = time.time()
            mu = np.empty(len(ytest))
            var = np.empty(len(ytest))
            chunk = H2_PREDICT_CHUNK if model_name == 'h2' else PREDICT_CHUNK
            for s in range(0, len(ytest), chunk):
                m, v = model.predict(xtest[s:s + chunk], tid=0)
                mu[s:s + chunk], var[s:s + chunk] = m[:, 0], v[:, 0]
            predict_time = time.time() - start
            noise_variance = hyperparameters['noise_variance'][0]
            mu += ymean
            var = np.maximum(var, 0.0) + noise_variance
            result = scores(ytest, mu, var)
            np.savez(os.path.join(directory, 'test_predictions_%s.npz' % tag), xtest=xtest, ytest=ytest, mu=mu, var=var)
            row = {'model': model_name, 'optimizer': optimizer, 'train_time': train_time, 'likelihood_evaluations': nfev,
                   'time_per_evaluation': train_time / max(nfev, 1), 'predict_time': predict_time,
                   'log_marginal_likelihood': model_stats['log_marginal_likelihood'], 'hyperparameters': hyperparameters,
                   'parameter_vector': model.M.get_parameter_vector(), **result}
            summary.append(row)
            print("=== %s, %s: training %.1f s (%d likelihood evaluations), prediction %.1f s; hyperparameters %s; "
                  % (model_name, optimizer, train_time, nfev, predict_time, hyperparameters)
                  + ", ".join("%s %.4f" % (k, result[k]) for k in ('MAE', 'RMSE', 'CRPS', 'INT', 'CVG')))
            write_summary(output_dir, dataset, summary, info)

if __name__ == "__main__":
    main()
