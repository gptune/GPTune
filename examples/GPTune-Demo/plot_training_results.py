#! /usr/bin/env python
"""
Plots of the GP hyperparameter trainings of model_comparison_updated_superlu.py, made only from the
files it saves for every run (sample count N, optimizer), so they can be remade without rerunning:
  training_metrics_obj<o>_N<N>_<optimizer>.csv  -loglikelihood, test RMSE and CRPS along the training
  run_stats_obj<o>_N<N>_<optimizer>.json        GPTune statistics of the run (times, evaluations, ...)

Usage: python plot_training_results.py [results directory] [objective]
makes training_history_obj<o>_N<N>.pdf/png for every N and optimizer_compare_obj<o>.pdf/png.
"""
import glob
import json
import os
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

OPTIMIZERS = ['gradient', 'finite difference', 'mcmc', 'mala'] # the model trainings plotted together
COLORS = {'gradient': 'blue', 'finite difference': 'red', 'mcmc': 'purple', 'mala': 'orange'}
NAMES = {'gradient': 'gradient L-BFGS', 'finite difference': 'finite difference L-BFGS', 'mcmc': 'random-walk MCMC', 'mala': 'MALA'}
# A marker per optimizer. The mcmc and mala runs stop at the same sampling-time cap, so their model times
# coincide: mcmc is drawn on top with hollow markers, so that the mala markers stay visible under them.
# 'size' scales the marker size of the plot.
STYLES = {'gradient': dict(marker='o', linestyle='-'),
          'finite difference': dict(marker='s', linestyle='-'),
          'mcmc': dict(marker='D', markerfacecolor='none', markeredgewidth=1.5, linestyle='--', zorder=3),
          'mala': dict(marker='^', linestyle='-', zorder=2)}
MODEL_LABEL = 'Sparse GP (george + SuperLU)' # the GP model named in the figure titles


def plot_style(optimizer, markersize):
    """matplotlib line and marker arguments of an optimizer, for a base marker size"""
    style = dict(STYLES[optimizer])
    return dict(style, markersize=markersize * style.pop('size', 1.0), color=COLORS[optimizer])


def run_file(prefix, objtype, N, optimizer, extension, directory='.'):
    return os.path.join(directory, '%s_obj%d_N%d_%s.%s' % (prefix, objtype, N, optimizer.replace(' ', '_'), extension))


def plot_training_histories(objtype, N, directory='.', ntest=1000):
    """
    -loglikelihood, test RMSE and test CRPS against the training time for every optimizer in
    OPTIMIZERS with a training_metrics file for this N. L-BFGS iterations are joined by lines; the
    best MCMC sample so far is drawn as a step function.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for optimizer in OPTIMIZERS:
        filename = run_file('training_metrics', objtype, N, optimizer, 'csv', directory)
        if not os.path.exists(filename):
            continue
        print("plotting the training history in", filename)
        rows = np.atleast_1d(np.genfromtxt(filename, delimiter=',', names=True))
        if optimizer in ('mcmc', 'mala'):
            label, drawstyle = "%s, best sample so far (%d evaluations)" % (NAMES[optimizer], rows['nfev'][-1]), 'steps-post'
        else:
            label, drawstyle = "%s (%d iterations)" % (NAMES[optimizer], len(rows) - 1), 'default'
        for ax, key in zip(axes, ['nll', 'rmse', 'crps']):
            done = np.isfinite(rows[key])
            ax.plot(rows['time'][done], rows[key][done], label=label, drawstyle=drawstyle, **plot_style(optimizer, 3))
    for ax, ylabel in zip(axes, ['-log likelihood', 'test RMSE (%d points)' % ntest, 'test CRPS (%d points)' % ntest]):
        ax.set_xlabel('training time (s)')
        ax.set_ylabel(ylabel)
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False)
    fig.suptitle('%s, objective %d, N=%d: hyperparameter training' % (MODEL_LABEL, objtype, N), fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(directory, 'training_history_obj%d_N%d.%s' % (objtype, N, extension)), dpi=150)
    plt.close(fig)


def load_run_stats(objtype, directory='.'):
    """{optimizer: sorted list of the run_stats of every N}"""
    runs = {}
    for filename in glob.glob(os.path.join(directory, 'run_stats_obj%d_N*_*.json' % objtype)):
        with open(filename) as f:
            stats = json.load(f)
        runs.setdefault(stats['optimizer'], []).append(stats)
    return {optimizer: sorted(r, key=lambda s: s['N']) for optimizer, r in runs.items()}


def plot_optimizer_scaling(objtype, directory='.'):
    """
    Model (training) time, search time, time per likelihood evaluation and number of likelihood
    evaluations against N for every optimizer, from the run_stats files.
    """
    runs = load_run_stats(objtype, directory)
    fontsize = 8
    plt.rcParams.update({'font.size': fontsize})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    panels = [(axes[0, 0], 'time_model', 'Model Time', 'Time (sec)'),
              (axes[0, 1], 'time_search', 'Search Time', 'Time (sec)'),
              (axes[1, 0], 'time_model_per_likelihoodeval', 'Model Time per Likelihood Evaluation', 'Time (sec)'),
              (axes[1, 1], 'modeling_iteration', 'Likelihood Evaluations', 'Evaluations')]
    for optimizer in OPTIMIZERS:
        if optimizer not in runs:
            continue
        Ns = [s['N'] for s in runs[optimizer]]
        for ax, key, _, _ in panels:
            values = [s['stats'][key][0] if isinstance(s['stats'][key], list) else s['stats'][key] for s in runs[optimizer]]
            ax.loglog(Ns, values, label=NAMES[optimizer], **plot_style(optimizer, 6))
    for ax, _, title, ylabel in panels:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel('Sample Count', fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1)
    fig.suptitle('%s hyperparameter training, objective %d' % (MODEL_LABEL, objtype), fontsize=fontsize + 1)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(directory, 'optimizer_compare_obj%d.%s' % (objtype, extension)), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use('Agg')
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    objtype = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Ns = sorted({int(re.search(r'_N(\d+)_', f).group(1)) for f in glob.glob(os.path.join(directory, 'training_metrics_obj%d_N*_*.csv' % objtype))})
    for N in Ns:
        plot_training_histories(objtype, N, directory)
    plot_optimizer_scaling(objtype, directory)
