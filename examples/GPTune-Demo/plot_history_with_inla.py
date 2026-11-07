#! /usr/bin/env python
"""
Training histories of the Wendland GP at N=100000 (the files in this directory: L-BFGS from the current
code, MCMC and MALA from the standalone runs) and of the INLA GP at N=102400 (job 59033221,
inla_vs_wendland/59033221/inla/), on the 2D Schwefel objective (objective 4), in one figure:
-loglikelihood, test RMSE and test CRPS against the training time (log scale, since the INLA trainings
take at most 190 s). Both drivers use the same 1000 test points (seed 2026).

Usage: python plot_history_with_inla.py   (writes training_history_obj4_N100000_with_inla.pdf/png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_training_results import OPTIMIZERS, NAMES, plot_style, run_file

OBJTYPE = 4
RUNS = [  # (model label, N, directory, line style of the INLA curves or None to keep the optimizer's style)
    ('Wendland', 100000, '.', None),
    ('INLA', 102400, os.path.join('inla_vs_wendland', '59033221', 'inla'), ':'),
]
NTEST = 1000


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for model, N, directory, linestyle in RUNS:
        for optimizer in OPTIMIZERS:
            rows = np.atleast_1d(np.genfromtxt(run_file('training_metrics', OBJTYPE, N, optimizer, 'csv', directory), delimiter=',', names=True))
            if optimizer in ('mcmc', 'mala'):
                what, drawstyle = "best sample so far, %d evaluations" % rows['nfev'][-1], 'steps-post'
            else:
                what, drawstyle = "%d iterations" % (len(rows) - 1), 'default'
            style = plot_style(optimizer, 3.5)
            if linestyle is not None:  # INLA: the optimizer's color and marker, hollow, on a dotted line
                style.update(linestyle=linestyle, markerfacecolor='white', markeredgewidth=1.2, linewidth=1.6)
            label = "%s (N = %s): %s (%s)" % (model, format(N, ','), NAMES[optimizer], what)
            for ax, key in zip(axes, ['nll', 'rmse', 'crps']):
                keep = np.isfinite(rows[key]) & (rows['time'] > 0)
                assert keep.sum() == np.isfinite(rows[key]).sum(), "a history entry at time 0 would be dropped by the log axis"
                ax.plot(rows['time'][keep], rows[key][keep], label=label, drawstyle=drawstyle, **style)
    for ax, ylabel in zip(axes, ['-log likelihood', 'test RMSE (%d points)' % NTEST, 'test CRPS (%d points)' % NTEST]):
        ax.set_xscale('log')
        ax.set_xlabel('training time (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='major', color='#e3e3e3', linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Sparse GP (george + SuperLU), objective 4 (2D Schwefel): Wendland and INLA hyperparameter training', fontsize=10)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    for extension in ('pdf', 'png'):
        fig.savefig('training_history_obj%d_N100000_with_inla.%s' % (OBJTYPE, extension), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
