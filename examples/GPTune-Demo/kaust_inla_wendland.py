#! /usr/bin/env python
"""
Sub-competition 2b, dataset 1 of the 2021 KAUST Competition on Spatial Statistics for Large Datasets with the two
sparse GP models of george with SuperLU_DIST, trained by GPTune's Model_George (the settings of the optimizers of
model_comparison_updated_superlu.py):
  wendland  the Wendland C2 function tapering a base kernel (sparse covariance matrix)
  inla      the INLA/SPDE Matern model on a lattice (sparse precision matrix, george/inla.py)
and with the dense RBF kernel compressed by butterflypack:
  h2        the H2 format on GPUs (run_spatial_h2_gpu.sbatch runs this driver with the butterflypack workers)
Other datasets of the same form (an npz with xtrain, ytrain, xtest, ytest, and optionally mtest, the noise-free
values at the testing locations, or xtest2 and ytest2, a second testing set scored separately) are given with
KAUST_DATA and GP_DATASET, e.g. the replica of the 2022 dataset 1b-1 of kaust2022_1b1_replica.py and the 2023
datasets of kaust2023_2b_npz.py.

The data are a zero-mean stationary isotropic Gaussian field on [0, 1]^2 with the Matern covariance
sigma^2 2^(1-nu) / Gamma(nu) (r/beta)^nu K_nu(r/beta), sigma^2 = 1.5, beta = 0.063240, nu = 1.5, no nugget, at
1,000,000 uniformly random locations: 900,000 training values and 100,000 testing values (released with the
competition datasets). In the Matern parameterization of george and of the INLA model,
M_nu(sqrt(2 nu) r / l), the true length scale is l = sqrt(2 nu) beta = 0.10953. The ranking criterion of the
competition was the RMSE of the predictions at the testing locations (best 9.47e-4, ExaGeoStat's exact kriging
being close to it). Both models are trained on the values minus their training mean, with one length scale per
dimension.

The predictive means are computed at all the testing locations: interpolated from the lattice for INLA, and
from the training values within the support of the Wendland function for Wendland. The predictive variances,
for the CRPS, the 95% interval score and coverage (of the observations: the noise variance is added), are
computed at the first KAUST_VAR_POINTS testing locations (for Wendland, they need a solve with a right-hand side
of the size of the training data per location).

The environment selects the runs (run_kaust_gpu.sbatch runs this driver with the SuperLU_DIST workers):
  KAUST_DATA        the data (default $PSCRATCH/kaust2021/kaust2021_2b_dataset01.npz: xtrain, ytrain, xtest,
                    ytest from dataset01_training.csv, dataset01_testing.csv and true_measurements/Z_01.csv)
  GP_MODELS         comma-separated models, each "wendland" or "inla" followed by ":key=value" settings
                    (default "wendland,inla"):
                      wendland: cutoff (largest support radius, default about 78 neighbours per sample),
                                base (RBF, Matern32 or Matern52, default Matern32: nu = 1.5)
                      inla:     shape (lattice nodes per dimension, default sqrt(N); auto: the grid of the
                                inputs), nu (default 1.5), buffer (the width of the lattice buffer, default
                                twice the largest length-scale maximum); sphere (the subdivision level of the icosahedral
                                mesh of the SPDE on the unit sphere, 3D inputs on the sphere; nu an integer,
                                default 1)
                      h2:       kern (RBF, Exp, Matern32 or Matern52, default RBF; all have GPU entry
                                evaluators; Exp is exp(-r), Matern with nu = 1/2)
                      any:      noisemin (the lower bound of the noise variance); iso=1 (one length scale for
                                all the dimensions: h2 and inla)
  GP_OPTIMIZERS     default "gradient,finite difference" (also "mcmc", "mala")
  GP_DATASET        the name of the dataset in the file names (default kaust_2b1)
  GP_NOISE, GP_AMPLITUDE, GP_LENGTHSCALE
                    "min max initial" ranges of the hyperparameters (linear scale; defaults 1e-6 1 1e-3,
                    1e-2 1e2 1, 1e-3 1 0.1)
  GP_LBFGS_MAXITER  the L-BFGS iteration limit; 0 for no optimization (prediction at the initial hyperparameters)
  GP_LENGTHSCALE_DIMS
                    per-dimension length-scale ranges replacing GP_LENGTHSCALE for anisotropic h2 and inla
                    models, "min max initial" per input dimension separated by ';' (e.g. space-time inputs:
                    "0.0025 1.5 0.06; 0.0025 1.5 0.06; 0.005 2 0.02")
  KAUST_SUBSAMPLE   use only this many training (and at most this many testing) values, for quick tests
  KAUST_VAR_POINTS  the number of testing locations of the predictive variances (default 2000; per testing set)
  GP_OUTPUT_DIR     the directory of the results (default .)
Writes <GP_DATASET>_compare.csv/json and, per model, the training histories and the test predictions.
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import BallTree

from pdbridge import superlu_factor, superlu_logdet, superlu_solve
# the sparse george models call superlu_factor/logdet/solve through george.solvers.basic
import george.solvers.basic
for _f in (superlu_factor, superlu_logdet, superlu_solve):
    setattr(george.solvers.basic, _f.__name__, _f)

from GPTune.options import Options
from GPTune.model import Model_George
from GPTune.data import Data

TRUTH = {'sigma2': 1.5, 'beta': 0.063240, 'nu': 1.5, 'tau2': 0.0} # of the 2021 2b dataset 1
VARIANCE_CHUNK = 200 # testing locations whose variances are computed together (the Wendland cross-covariance is dense)
MEAN_CHUNK = 64 # testing locations of a block of the dense RBF cross-covariance of the H2 predictive means


def parse_model(spec):
    """'inla:shape=1024:nu=1.5' -> ('inla', {'shape': 1024, 'nu': 1.5}, 'inla_shape1024_nu1.5')"""
    parts = spec.split(':')
    settings = {}
    for part in parts[1:]:
        key, value = part.split('=')
        settings[key] = value if key in ('base', 'kern') or value == 'auto' else float(value)
    name = '_'.join([parts[0]] + ['%s%s' % (k, v if isinstance(v, str) else '%g' % v) for k, v in settings.items()])
    return parts[0], settings, name


def hyperparameter_range(variable, default):
    """[min, max, initial] of a hyperparameter: the environment variable variable, or default."""
    return [float(v) for v in os.environ[variable].split()] if os.environ.get(variable) else list(default)


def scores(y, mu, var):
    """MAE and RMSE, and, where var is finite, CRPS, 95% interval score and coverage."""
    out = {'MAE': np.mean(np.abs(y - mu)), 'RMSE': np.sqrt(np.mean((y - mu) ** 2))}
    known = np.isfinite(var)
    y, mu, sd = y[known], mu[known], np.sqrt(var[known])
    z = (y - mu) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    alpha = 0.05
    low, high = mu - norm.ppf(1 - alpha / 2) * sd, mu + norm.ppf(1 - alpha / 2) * sd
    interval = (high - low) + 2 / alpha * (low - y) * (y < low) + 2 / alpha * (y - high) * (y > high)
    out.update({'CRPS': np.mean(crps), 'INT': np.mean(interval), 'CVG': np.mean((y >= low) & (y <= high)),
                'variance_points': int(known.sum())})
    return out


def model_options(kind, settings, optimizer, ntrain, history_file, lbfgs_times):
    """The Model_George options of a model and an optimizer."""
    options = Options()
    noise = hyperparameter_range('GP_NOISE', [1e-6, 1.0, 1e-3])
    if 'noisemin' in settings:
        noise[0] = settings['noisemin']
        noise[2] = max(noise[2], 10 * noise[0])
    if os.environ.get('GP_LBFGS_MAXITER'):
        # 0: no optimization, the prediction at the initial hyperparameters (e.g. the last iterate of a run that
        # timed out, pinned by the ranges)
        options.update(model_lbfgs_maxiter=int(os.environ['GP_LBFGS_MAXITER']))
    if os.environ.get('GP_LENGTHSCALE_DIMS'):
        options.update(model_lengthscale_dims=[[float(v) for v in dim.split()]
                                               for dim in os.environ['GP_LENGTHSCALE_DIMS'].split(';') if dim.strip()])
    options.update(model_class='Model_George', model_noisevariance=noise,
                   model_amplitude=hyperparameter_range('GP_AMPLITUDE', [1e-2, 1e2, 1.0]),
                   model_lengthscale=hyperparameter_range('GP_LENGTHSCALE', [1e-3, 1.0, 0.1]), model_random_seed=0,
                   verbose=True, debug=False, model_history_file=history_file)
    if kind == 'h2':
        # the dense RBF, Matern-3/2 or Matern-5/2 kernel (kern; all three have GPU entry evaluators in george) in
        # butterflypack's H2 format, with the compression settings of the workers (run_spatial_h2_gpu.sbatch); one
        # length scale per dimension unless iso=1
        options.update(model_bpack=True, model_kern=settings.get('kern', 'RBF'), model_isotropic=settings.get('iso', 0) == 1)
    elif kind == 'wendland':
        # about 78 neighbours per sample by default: the cutoff of model_comparison_updated_superlu.py
        options.update(model_sparse=True, model_kern='WendlandC2', model_wendland_base=settings.get('base', 'Matern32'),
                       model_cutoff=settings.get('cutoff', 0.0158 * np.sqrt(100000 / ntrain)))
    elif kind == 'inla':
        # shape=auto: the lattice of the data (inputs on a grid, e.g. the 0.125 degree MODIS grid)
        shape = settings.get('shape', round(np.sqrt(ntrain)))
        options.update(model_kern='INLA', model_inla_nu=settings.get('nu', 1.5),
                       model_inla_shape=None if shape == 'auto' else int(shape))
        if 'buffer' in settings:
            # the width of the lattice buffer (default twice the largest length-scale maximum)
            options.update(model_inla_buffer=settings['buffer'])
        if 'sphere' in settings:
            options.update(model_inla_sphere=int(settings['sphere']), model_inla_nu=settings.get('nu', 1.0), model_isotropic=True)
    else:
        raise ValueError("unknown model %s" % kind)
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


def wendland_mean(model, xtest, cutoff):
    """
    The predictive means of a Wendland model at xtest (without the training mean): sum_j k(x, x_j) alpha_j over
    the training locations x_j within the support radius of x, alpha = K^-1 y.
    """
    gp = model.M
    alpha = gp._compute_alpha(np.ravel(model.y), True)
    neighbours = BallTree(gp._x).query_radius(xtest, r=cutoff)
    mu = np.zeros(len(xtest))
    for i, idx in enumerate(neighbours):
        if len(idx):
            mu[i] = gp.kernel.get_value(xtest[i:i + 1], gp._x[idx])[0] @ alpha[idx]
    return mu


# k(r^2) of the stationary kernels of the h2 model (george's definitions), in place on an array of r^2
def _rbf_profile(d2):
    np.exp(-0.5 * d2, out=d2)
    return d2


def _exp_profile(d2):
    np.sqrt(d2, out=d2)
    np.exp(-d2, out=d2)
    return d2


def _matern32_profile(d2):
    a = np.sqrt(3.0 * d2)
    return np.multiply(1.0 + a, np.exp(-a), out=d2)


def _matern52_profile(d2):
    a = np.sqrt(5.0 * d2)
    return np.multiply(1.0 + a + a * a / 3.0, np.exp(-a), out=d2)


KERNEL_PROFILES = {'ExpSquaredKernel': _rbf_profile, 'ExpKernel': _exp_profile, 'Matern32Kernel': _matern32_profile,
                   'Matern52Kernel': _matern52_profile}


def kernel_mean(model, xtest, threads=32):
    """
    The predictive means of an h2 model (a constant times an axis-aligned ExpSquaredKernel, ExpKernel,
    Matern32Kernel or Matern52Kernel) at xtest (without the training mean): K(xtest, X) alpha, alpha = K^-1 y solved with the model's
    solver, with the dense cross-covariance formed by blocks of MEAN_CHUNK testing locations in threads.
    """
    gp = model.M
    alpha = gp._compute_alpha(np.ravel(model.y), True)
    kernel = gp.kernel
    parts = [kernel.k1, kernel.k2]
    constant = [k for k in parts if isinstance(k, george.kernels.ConstantKernel)][0]
    squared = [k for k in parts if type(k).__name__ in KERNEL_PROFILES][0]
    profile = KERNEL_PROFILES[type(squared).__name__]
    amplitude = constant.get_value(np.zeros((1, constant.ndim)))[0, 0]
    scale = 1 / np.sqrt(np.diag(squared.metric.to_matrix())) # (axis-aligned metric over all the dimensions)
    a, b = xtest * scale, gp._x * scale
    bb = np.sum(b * b, axis=1)

    def block(s):
        ab = a[s:s + MEAN_CHUNK]
        d2 = np.sum(ab * ab, axis=1)[:, None] + bb[None, :] - 2 * ab @ b.T
        np.maximum(d2, 0, out=d2)
        return amplitude * (profile(d2) @ alpha)

    # (checked against george's kernel on a few entries)
    k_check = kernel.get_value(xtest[:8], gp._x[:1000])
    d2 = np.sum((a[:8, None, :] - b[None, :1000, :]) ** 2, axis=2)
    error = np.max(np.abs(k_check - amplitude * profile(d2))) / np.max(np.abs(k_check))
    if error > 1e-10:
        raise RuntimeError("kernel_mean: kernel mismatch %.2e" % error)
    with ThreadPoolExecutor(threads) as pool:
        return np.concatenate(list(pool.map(block, range(0, len(xtest), MEAN_CHUNK))))


def write_summary(output_dir, summary, info):
    """<dataset>_compare.json/csv and the table of the runs done so far."""
    prefix = os.path.join(output_dir, '%s_compare' % info['dataset'])
    with open(prefix + '.json', 'w') as f:
        json.dump(dict(info, runs=summary), f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    keys = ['train_time', 'likelihood_evaluations', 'time_per_evaluation', 'predict_time', 'RMSE', 'MAE', 'CRPS', 'INT', 'CVG',
            'sigma2', 'beta_x', 'beta_y', 'tau2']
    extra = [k for k in ('RMSE_mean', 'RMSE_2', 'MAE_2', 'CRPS_2', 'INT_2', 'CVG_2') if summary and k in summary[0]]
    with open(prefix + '.csv', 'w') as f:
        f.write('model,optimizer,' + ','.join(keys + extra) + '\n')
        for row in summary:
            f.write('%s,%s,' % (row['model'], row['optimizer']) + ','.join('%.6g' % row[k] for k in keys + extra) + '\n')
    if info['dataset'] == 'kaust_2b1':
        print("KAUST 2021 2b dataset 1, %d training and %d testing values (true sigma^2 %g, beta %g, nu %g, tau^2 %g):"
              % (info['ntrain'], info['ntest'], TRUTH['sigma2'], TRUTH['beta'], TRUTH['nu'], TRUTH['tau2']))
    else:
        print("%s, %d training and %d testing values%s%s:" % (info['dataset'], info['ntrain'], info['ntest'],
              " (RMSE_mean: of the noise-free values)" if 'RMSE_mean' in extra else "",
              " (_2: the second testing set)" if 'RMSE_2' in extra else ""))
    print("  %-28s %-18s %9s %6s %8s %8s %10s %9s %8s %8s %6s %7s %8s %8s %9s"
          % ('model', 'optimizer', 'train (s)', 'evals', 'per eval', 'pred (s)', 'RMSE', 'MAE', 'CRPS', 'INT', 'CVG',
             'sigma^2', 'beta_x', 'beta_y', 'tau^2'))
    for row in summary:
        print("  %-28s %-18s %9.1f %6d %8.3f %8.1f %10.4e %9.3e %8.2e %8.2e %6.3f %7.3f %8.5f %8.5f %9.2e"
              % tuple([row['model'], row['optimizer']] + [row[k] for k in keys])
              + ''.join("  %s %.4e" % (k, row[k]) for k in extra))


def main():
    data_file = os.environ.get('KAUST_DATA', os.path.join(os.environ.get('PSCRATCH', '.'), 'kaust2021/kaust2021_2b_dataset01.npz'))
    models = os.environ.get('GP_MODELS', 'wendland,inla').split(',')
    optimizers = os.environ.get('GP_OPTIMIZERS', 'gradient,finite difference').split(',')
    subsample = int(os.environ['KAUST_SUBSAMPLE']) if os.environ.get('KAUST_SUBSAMPLE') else None
    var_points = int(os.environ.get('KAUST_VAR_POINTS', 2000))
    output_dir = os.environ.get('GP_OUTPUT_DIR', '.')
    dataset = os.environ.get('GP_DATASET', 'kaust_2b1')

    z = np.load(data_file)
    xtrain, ytrain, xtest, ytest = z['xtrain'], z['ytrain'], z['xtest'], z['ytest']
    mtest = z['mtest'] if 'mtest' in z else None
    # a second testing set (the 2023 competition: testing set 2, uniform locations), scored separately
    xtest2, ytest2 = (z['xtest2'], z['ytest2']) if 'xtest2' in z else (None, None)
    if subsample is not None:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(ytrain), subsample, replace=False)
        xtrain, ytrain = xtrain[keep], ytrain[keep]
        xtest, ytest = xtest[:subsample], ytest[:subsample]
        mtest = None if mtest is None else mtest[:subsample]
        if xtest2 is not None:
            xtest2, ytest2 = xtest2[:subsample], ytest2[:subsample]
    ymean = np.mean(ytrain)
    var_points = min(var_points, len(ytest))
    print("%s (%s): %d training and %d testing values (variances at %d), training mean %.5f"
          % (dataset, data_file, len(ytrain), len(ytest), var_points, ymean))
    data = Data(problem=None)
    data.I, data.P, data.O = [[0]], [xtrain], [(ytrain - ymean)[:, None]]
    info = {'dataset': dataset, 'data': data_file, 'ntrain': len(ytrain), 'ntest': len(ytest), 'variance_points': var_points,
            'training_mean': ymean, 'truth': TRUTH if dataset == 'kaust_2b1' else None,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID')}

    summary = []
    for spec in models:
        kind, settings, name = parse_model(spec)
        directory = os.path.join(output_dir, name)
        os.makedirs(directory, exist_ok=True)
        lbfgs_times = []
        for optimizer in optimizers:
            tag = '%s_%s' % (dataset, optimizer.replace(' ', '_'))
            options = model_options(kind, settings, optimizer, len(ytrain),
                                    os.path.join(directory, 'training_iterations_%s.csv' % tag), lbfgs_times)
            print("=== %s, %s: %s" % (name, optimizer, {k: options[k] for k in ('model_kern', 'model_grad', 'model_mcmc', 'model_mcmc_max_time',
                                                                              'model_inla_shape', 'model_inla_nu', 'model_inla_sphere', 'model_cutoff',
                                                                              'model_wendland_base') if k in options}))
            model = Model_George(problem=None, computer=None)
            start = time.time()
            hyperparameters, modeling_options, model_stats, nfev = model.train(data, **options)
            train_time = time.time() - start
            if optimizer in ('gradient', 'finite difference'):
                lbfgs_times.append(train_time)

            # predictive means at all the testing locations, variances at the first var_points ones, for each testing set
            noise_variance = hyperparameters['noise_variance'][0]

            def predict_scores(xt, yt, mt):
                if kind == 'inla':
                    mu = model.M.predict(np.ravel(model.y), xt, return_cov=False)
                elif kind == 'h2':
                    mu = kernel_mean(model, xt)
                else:
                    mu = wendland_mean(model, xt, hyperparameters['rc'][0])
                var = np.full(len(yt), np.nan)
                npoints = min(var_points, len(yt))
                for s in range(0, npoints, VARIANCE_CHUNK):
                    e = min(s + VARIANCE_CHUNK, npoints)
                    m, v = model.predict(xt[s:e], tid=0)
                    mu_check = m[:, 0] - mu[s:e]
                    if np.max(np.abs(mu_check)) > 1e-6 * max(1.0, np.max(np.abs(mu[s:e]))):
                        print("warning: predictive means of the variance chunk differ by %.2e" % np.max(np.abs(mu_check)))
                    var[s:e] = np.maximum(v[:, 0], 0.0)
                mu = mu + ymean
                var = var + noise_variance
                result = scores(yt, mu, var)
                if mt is not None:
                    result['RMSE_mean'] = np.sqrt(np.mean((mt - mu) ** 2))
                return result, mu, var

            start = time.time()
            result, mu, var = predict_scores(xtest, ytest, mtest)
            np.savez(os.path.join(directory, 'test_predictions_%s.npz' % tag), xtest=xtest, ytest=ytest, mu=mu, var=var)
            if xtest2 is not None:
                result2, mu2, var2 = predict_scores(xtest2, ytest2, None)
                result.update({k + '_2': v for k, v in result2.items()})
                np.savez(os.path.join(directory, 'test2_predictions_%s.npz' % tag), xtest=xtest2, ytest=ytest2, mu=mu2, var=var2)
            predict_time = time.time() - start

            # the estimated Matern parameters in the parameterization of the competition: beta = l / sqrt(2 nu)
            if kind == 'inla':
                nu = settings.get('nu', 1.0 if 'sphere' in settings else 1.5)
            elif kind == 'h2':
                nu = {'Exp': 0.5, 'Matern32': 1.5, 'Matern52': 2.5}.get(settings.get('kern', 'RBF'), np.inf)
            else:
                nu = {'Matern32': 1.5, 'Matern52': 2.5}.get(settings.get('base', 'Matern32'), np.inf)
            lengthscale = np.array(hyperparameters['lengthscale'], dtype=float)
            beta = lengthscale / np.sqrt(2 * nu) if np.isfinite(nu) else lengthscale
            row = {'model': name, 'optimizer': optimizer, 'train_time': train_time, 'likelihood_evaluations': nfev,
                   'time_per_evaluation': train_time / max(nfev, 1), 'predict_time': predict_time,
                   'log_marginal_likelihood': model_stats['log_marginal_likelihood'], 'hyperparameters': hyperparameters,
                   'parameter_vector': model.M.get_parameter_vector(), 'sigma2': hyperparameters['variance'][0],
                   'beta_x': beta[0], 'beta_y': beta[-1], 'tau2': noise_variance, 'settings': settings, **result}
            summary.append(row)
            print("=== %s, %s: training %.1f s (%d likelihood evaluations), prediction %.1f s; hyperparameters %s; "
                  % (name, optimizer, train_time, nfev, predict_time, hyperparameters)
                  + ", ".join("%s %.4e" % (k, result[k]) for k in ('RMSE', 'MAE', 'CRPS', 'INT', 'CVG')))
            write_summary(output_dir, summary, info)


if __name__ == "__main__":
    main()
