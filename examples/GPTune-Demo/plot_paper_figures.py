#!/usr/bin/env python3
"""
Figures 2-5 of the comparative-study paper, drawn at their printed size from the saved run files:

  rank_vs_density.pdf       factorization rank of every build against N l^d (plot_rank_vs_density.py reads
                            the ranks from the worker logs and the hyperparameters from the driver logs):
                            (a) objective 1 (oscillatory 1D) on CPUs, unrestricted length scale,
                            scal_obj1_h2cpu and scal_obj1_hodlrcpu; (b) objective 2 (2D isotropic) on GPUs,
                            length scale >= 0.01, scal_h2gpu_2d_iso and scal_hodlrgpu_2d_iso
  scaling_sparse.pdf        Wendland and INLA on the 2D Schwefel objective (objective 4), 16 GPUs:
                            training time and time per likelihood evaluation against N, four optimizers
                            (Wendland: scaling_obj4_reuse/, INLA: inla_scaling_obj4/inla/)
  history_100k_wendland_inla.pdf
                            -log likelihood, test RMSE and test CRPS along the trainings of the Wendland
                            GP at N=100,000 (training_metrics files in this directory) and the INLA GP at
                            N=102,400 (inla_vs_wendland/59033221/inla/); the same data as
                            plot_history_with_inla.py
  h2_vs_hodlr_1d_cpu.pdf    H2 against HODLR on CPUs, objective 1 (oscillatory 1D), time per likelihood
                            evaluation against N. H2: scal_obj1_h2cpu (unrestricted length scale) with
                            scal_obj1ls_h2cpu (length scale in [0.02, 0.03]) taking precedence where both
                            have the run, i.e. from 1.6M on; HODLR: scal_obj1ls_hodlrcpu (restricted
                            throughout); the same data as hodlr_vs_h2_obj1_ls.pdf of plot_hodlr_vs_h2.py

Text is set at its printed size (8 pt, STIX serif like the paper's Times, Computer Modern math), with
TrueType fonts in the PDFs.  The four optimizers keep one color and one marker in every figure; the
palette passes colorblind (deuteranopia, protanopia, tritanopia) separation over all pairs and 3:1
contrast against white.

    python plot_paper_figures.py [output directory] [--png]
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from plot_rank_vs_density import collect as rank_builds

HERE = os.path.dirname(os.path.abspath(__file__))
CMP = os.path.join(HERE, "h2_unstructured_cmp")

RC = {"font.size": 8, "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "cm",
      "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.labelsize": 8, "axes.titlesize": 8.5,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
      "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
      "lines.linewidth": 1.2, "lines.markersize": 3.6, "legend.handlelength": 2.2,
      "savefig.bbox": "tight", "savefig.pad_inches": 0.02}
GRID = dict(color="#e4e4e4", linewidth=0.5)

# optimizer: (legend label, color, marker); MCMC is drawn hollow and on top, since its time per step
# is that of a likelihood evaluation and its points fall on the finite-difference ones
OPTIMIZERS = ["gradient", "finite difference", "mcmc", "mala"]
STYLE = {"gradient": ("gradient L-BFGS", "#1f5fb0", "o"),
         "finite difference": ("FD L-BFGS", "#e07020", "s"),
         "mcmc": ("MCMC", "#2a9d8f", "D"),
         "mala": ("MALA", "#a8327d", "^")}


def line_style(optimizer, hollow=False, linestyle="-"):
    label, color, marker = STYLE[optimizer]
    style = dict(color=color, marker=marker, linestyle=linestyle, markeredgewidth=0.9)
    if optimizer == "mcmc":
        style.update(markersize=3.2, zorder=3)
    if hollow or optimizer == "mcmc":
        style.update(markerfacecolor="white")
    return style


def optimizer_handles(optimizers=OPTIMIZERS):
    return [Line2D([], [], label=STYLE[o][0], **line_style(o)) for o in optimizers]


def first(value):
    return value[0] if isinstance(value, list) else value


def run_stats(directories, objtype, fmt=None, recursive=True):
    """{optimizer: {N: stats}} from run_stats_obj<o>_N<N>_<optimizer>.json under the directories (or, with
    recursive=False, directly in them); a later directory overrides an earlier one for the same optimizer and N."""
    runs = {}
    pattern = os.path.join("**", "run_stats_obj%d_N*_*.json" % objtype) if recursive else "run_stats_obj%d_N*_*.json" % objtype
    paths = [sorted(glob.glob(os.path.join(d, pattern), recursive=recursive)) for d in directories]
    if not any(paths):
        raise SystemExit("no run_stats_obj%d files under %s" % (objtype, ", ".join(directories)))
    for directory_paths in paths:
        for path in directory_paths:
            run = json.load(open(path))
            if fmt is not None and int(run.get("format", 0)) != fmt:
                continue
            n = int(re.search(r"_N(\d+)_", os.path.basename(path)).group(1))
            stats = run["stats"]
            runs.setdefault(run["optimizer"], {})[n] = {
                "time_model": first(stats["time_model"]),
                "time_per_eval": first(stats["time_model_per_likelihoodeval"]),
                "evaluations": first(stats["modeling_iteration"])}
    return runs


def log_axes(ax, xminor=True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", **GRID)
    ax.set_axisbelow(True)
    if not xminor:
        ax.xaxis.set_minor_formatter(NullFormatter())


def draw_series(ax, runs, key, optimizers=OPTIMIZERS, **extra):
    for optimizer in optimizers:
        series = runs.get(optimizer, {})
        counts = [n for n in sorted(series) if series[n][key] is not None]
        if counts:
            style = line_style(optimizer, **extra)
            ax.plot(counts, [series[n][key] for n in counts], **style)


def save(fig, out_dir, name, png):
    path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path)
    if png:
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", path)


def figure_scaling_sparse(out_dir, png):
    """Training time and time per likelihood evaluation against N, Wendland (left) and INLA (right)."""
    # Wendland: MCMC and MALA are the capped runs of 27 Sep (as in Table III); the uncapped MALA reruns in
    # scaling_obj4_mala_uncapped have no run_stats (the 100k one was stopped by its wall time, see its
    # timeout_6h_61634303/step700_summary.txt) and its subdirectories hold superseded attempts, so it is not read
    models = [("(a) Wendland GP, sparse $K$", run_stats([os.path.join(HERE, "scaling_obj4_reuse")], 4)),
              ("(b) INLA GP, sparse $Q_c$", run_stats([os.path.join(HERE, "inla_scaling_obj4", "inla")], 4))]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.7), sharex=True, sharey="row")
    for col, (title, runs) in enumerate(models):
        for row, key in enumerate(("time_model", "time_per_eval")):
            ax = axes[row, col]
            log_axes(ax)
            draw_series(ax, runs, key)
        axes[0, col].set_title(title, pad=4)
        axes[1, col].set_xlabel("number of training points $N$")
    axes[0, 0].set_ylabel("training time (s)")
    axes[1, 0].set_ylabel("time per evaluation (s)")
    axes[1, 0].set_xticks([1e5, 2e5, 4e5, 8e5, 1.6e6])
    axes[1, 0].set_xticklabels(["100k", "200k", "400k", "800k", "1.6M"])
    axes[1, 0].xaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].set_xlim(8.5e4, 1.9e6)
    for ax in axes[:, 0]:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    fig.tight_layout(h_pad=0.6, w_pad=1.2, rect=(0, 0, 1, 0.93))
    fig.legend(handles=optimizer_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, columnspacing=1.6)
    save(fig, out_dir, "scaling_sparse", png)


def training_metrics(directory, N, optimizer):
    path = os.path.join(directory, "training_metrics_obj4_N%d_%s.csv" % (N, optimizer.replace(" ", "_")))
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def figure_history(out_dir, png):
    """-log likelihood, test RMSE and test CRPS against the training time, Wendland and INLA at ~100k.
    Both MALA curves are uncapped runs with the default stagnation stop: Wendland = the 100k run of Table III
    (tab:wend_opt, job 61694865), INLA = the MALA row of Table IV (tab:schwefel_opt, job 61707211), 40 replayed
    history entries each; the other curves are the runs in HERE and inla_vs_wendland/59033221/inla."""
    models = [("Wendland, $N$ = 100,000", HERE, 100000, False, "-",
               {"mala": os.path.join(HERE, "scaling_obj4_mala_stagnation")}),
              ("INLA, $N$ = 102,400", os.path.join(HERE, "inla_vs_wendland", "59033221", "inla"), 102400, True, ":",
               {"mala": os.path.join(HERE, "inla_vs_wendland", "mala_stagnation", "inla")})]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for label, directory, N, hollow, linestyle, run_dirs in models:
        for optimizer in OPTIMIZERS:
            rows = training_metrics(run_dirs.get(optimizer, directory), N, optimizer)
            sampler = optimizer in ("mcmc", "mala")
            style = line_style(optimizer, hollow=hollow, linestyle=linestyle)
            style.update(markersize=2.6, linewidth=1.0)
            for ax, key in zip(axes, ("nll", "rmse", "crps")):
                keep = np.isfinite(rows[key]) & (rows["time"] > 0)
                assert keep.sum() == np.isfinite(rows[key]).sum(), "an entry at time 0 would vanish on the log axis"
                ax.plot(rows["time"][keep], rows[key][keep], drawstyle="steps-post" if sampler else "default", **style)
    for ax, ylabel in zip(axes, ("negative log-likelihood", "test RMSE (1,000 points)", "test CRPS (1,000 points)")):
        ax.set_xscale("log")
        ax.set_xlabel("training time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", **GRID)
        ax.set_axisbelow(True)
        ax.set_xlim(7e-3, 6e3)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    for ax in axes[1:]:
        ax.set_yscale("log")
        ax.set_ylim(8e-5, 1.2)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axes[0].yaxis.set_major_formatter(formatter)
    # direct labels for the two models; the line style and fill say the same in the legend
    text = dict(fontsize=7.5, color="#333333", ha="left")
    axes[0].set_ylim(-6.4e5, 1.9e5)
    axes[0].text(0.012, 1.2e5, "Wendland", va="bottom", **text)
    axes[0].text(0.012, -4.2e5, "INLA", va="bottom", **text)
    # INLA's RMSE stays flat near 1.4e-4 (label above it), its CRPS falls from 4.8e-4 (label below)
    for ax, inla_y in zip(axes[1:], (2.0e-4, 1.0e-4)):
        ax.text(0.012, 0.32, "Wendland", va="bottom", **text)
        ax.text(0.012, inla_y, "INLA", va="bottom", **text)
    handles = optimizer_handles() + [
        Line2D([], [], color="#555555", marker="o", markersize=3, linestyle="-", label=models[0][0]),
        Line2D([], [], color="#555555", marker="o", markersize=3, markerfacecolor="white", linestyle=":",
               label=models[1][0])]
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False,
               columnspacing=1.4, handlelength=2.4)
    save(fig, out_dir, "history_100k_wendland_inla", png)


def figure_h2_hodlr_cpu(out_dir, png):
    """Time per likelihood evaluation against N for H2 and HODLR on CPUs, objective 1."""
    h2 = run_stats([os.path.join(CMP, "scal_obj1_h2cpu"), os.path.join(CMP, "scal_obj1ls_h2cpu")], 1, fmt=7)
    hodlr = run_stats([os.path.join(CMP, "scal_obj1ls_hodlrcpu")], 1, fmt=1)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    log_axes(ax)
    draw_series(ax, hodlr, "time_per_eval", hollow=True, linestyle="--")
    draw_series(ax, h2, "time_per_eval")
    ax.set_xlabel("number of training points $N$")
    ax.set_ylabel("time per evaluation (s)")
    ticks = [1e5, 4e5, 1.6e6, 6.4e6, 2.56e7]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["100k", "400k", "1.6M", "6.4M", "25.6M"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(8e4, 3.2e7)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    two_row_format_legend(ax)
    fig.tight_layout()
    save(fig, out_dir, "h2_vs_hodlr_1d_cpu", png)


def format_handles():
    """legend entries of the two formats: H2 solid with filled markers, HODLR dashed with hollow ones"""
    return [Line2D([], [], color="#555555", marker="o", markersize=3, linestyle="-", label="$\\mathcal{H}^2$"),
            Line2D([], [], color="#555555", marker="o", markersize=3, markerfacecolor="white", linestyle="--",
                   label="HODLR")]


def two_row_format_legend(ax):
    """the optimizers in one row above a one-column plot, the two formats below them (columns fill first)"""
    o, formats = optimizer_handles(), format_handles()
    blank = Line2D([], [], linestyle="none", label=" ")
    handles = [o[0], formats[0], o[1], formats[1], o[2], blank, o[3], blank]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              columnspacing=0.9, handlelength=2.2, handletextpad=0.5, borderaxespad=0.2)


def table_runs(rows, objtype, fmt):
    """{optimizer: {N: stats}} for rows of Table IX (tab:h2gpu), given as [(run directory, [N, ...])] or
    [(run directory, [N, ...], optimizers to leave out)]: each N from the directory named with it, and only the
    run_stats files directly in the directory (failed attempts were moved to its subdirectories)"""
    runs = {}
    for directory, counts, *skip in rows:
        found = run_stats([os.path.join(CMP, directory)], objtype, fmt=fmt, recursive=False)
        for optimizer, series in found.items():
            if skip and optimizer in skip[0]:
                continue
            for n in counts:
                if n in series:
                    runs.setdefault(optimizer, {})[n] = series[n]
    return runs


def gpu_axes(ax, ticks, labels, xlim):
    log_axes(ax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_xlabel("number of training points $N$")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))


def figure_h2_hodlr_gpu_2d(out_dir, png):
    """Time per likelihood evaluation of H2 and HODLR on GPUs, the 2D rows of Table IX and the larger H2 runs:
    isotropic kernel (objective 2) and anisotropic kernel (objective 4, scaled-coordinate tree; H2 at 400k and
    800k from the earlier raw-coordinate runs, marked with a dagger as in the table). H2 to 800k ran ButterflyPACK
    3f05cd5 (iso) and eef6ed0/eb5e78b (scaled), 1.6M ran 1d3d6d1 (submit_h2_large.sh, no MALA)."""
    panels = [("(a) isotropic kernel", 2,
               [("scal_h2gpu_2d_iso", [100000, 200000, 400000, 800000, 1600000])],
               [("scal_hodlrgpu_2d_iso", [100000, 200000])], ["100k", "200k", "400k", "800k", "1.6M"]),
              ("(b) anisotropic kernel", 4,
               # (1.6M without gradient L-BFGS: its derivative compression for the long length scale took
               #  151 s per evaluation (median) against 1.4 s at 200k, a ButterflyPACK slowdown, 20 Nov 2026)
               [("scal_h2gpu_2d_aniso_scaled", [100000, 200000]),
                ("scal_h2gpu_2d_aniso_scaled", [1600000], ("gradient",)),
                ("scal_h2gpu_2d_aniso", [400000, 800000])],
               [("scal_hodlrgpu_2d_aniso_scaled", [100000, 200000])],
               ["100k", "200k", "400k\u2020", "800k\u2020", "1.6M"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (title, objtype, h2_rows, hodlr_rows, labels) in zip(axes, panels):
        draw_series(ax, table_runs(hodlr_rows, objtype, 1), "time_per_eval", hollow=True, linestyle="--")
        draw_series(ax, table_runs(h2_rows, objtype, 7), "time_per_eval")
        gpu_axes(ax, [1e5, 2e5, 4e5, 8e5, 1.6e6], labels, (8.5e4, 1.9e6))
        ax.set_title(title, pad=4)
    axes[0].set_ylabel("time per evaluation (s)")
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    fig.legend(handles=optimizer_handles() + format_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, columnspacing=1.6)
    save(fig, out_dir, "h2_vs_hodlr_2d_gpu", png)


def figure_h2_hodlr_gpu_3d(out_dir, png):
    """Time per likelihood evaluation of H2 and HODLR on GPUs, the 3D anisotropic rows of Table IX
    (objective 5): HODLR at 25k only, H2 at 50k (1d3d6d1, submit_aniso_scaled.sh) when present, 100k, 400k
    (1d3d6d1, submit_h2_large.sh) and, from the earlier raw-coordinate run, 200k."""
    h2 = table_runs([("scal_h2gpu_3d_aniso_scaled", [50000, 100000, 400000]), ("scal_h2gpu_3d_aniso", [200000])],
                    5, 7)
    hodlr = table_runs([("scal_hodlrgpu_3d_aniso_scaled", [25000])], 5, 1)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    draw_series(ax, hodlr, "time_per_eval", hollow=True, linestyle="--")
    draw_series(ax, h2, "time_per_eval")
    gpu_axes(ax, [2.5e4, 5e4, 1e5, 2e5, 4e5], ["25k", "50k", "100k", "200k\u2020", "400k"], (2.1e4, 4.7e5))
    ax.set_ylabel("time per evaluation (s)")
    two_row_format_legend(ax)
    fig.tight_layout()
    save(fig, out_dir, "h2_vs_hodlr_3d_gpu", png)


def figure_rank(out_dir, png):
    """Factorization rank against N l^d for H2 and HODLR: 1D oscillatory on CPUs, 2D isotropic on GPUs."""
    # (title, objective, run directories, which worker logs, rank axis range): the legends go top right, which
    # the builds leave empty (in 1D with headroom above the HODLR peak: no build reaches rank 25 beyond N l = 10^3).
    # 1D: every run, without scal_obj1_h2cpu/worker_N1600000.log, a rerun of the 1.6M MCMC run with the same
    # builds; 2D: the runs of Table IX (100k-400k), i.e. the optimizer logs worker_N<n>_<optimizer>.log
    one_each = lambda w: not w.endswith(os.path.join("scal_obj1_h2cpu", "worker_N1600000.log"))
    table9 = lambda w: re.search(r"worker_N(\d+)_[a-z]+\.log$", w) is not None and int(re.search(r"_N(\d+)_", w).group(1)) <= 400000
    panels = [("(a) 1D oscillatory objective, CPU", 1, ["scal_obj1_h2cpu", "scal_obj1_hodlrcpu"], one_each, (1.5, 450)),
              ("(b) 2D isotropic objective, GPU", 2, ["scal_h2gpu_2d_iso", "scal_hodlrgpu_2d_iso"], table9, (8, 1.6e4))]
    # HODLR first, as filled dots, and H2 on top as crosses, so that both stay visible where they overlap
    formats = [(1, "HODLR", "#e07020", dict(marker="o", s=7, linewidths=0, alpha=0.45)),
               (7, "$\\mathcal{H}^2$", "#1f5fb0", dict(marker="x", s=9, linewidths=0.6, alpha=0.6))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (title, objtype, directories, keep, ylim) in zip(axes, panels):
        points = rank_builds([os.path.join(CMP, d) for d in directories], objtype, "factor", keep=keep)
        for fmt, name, color, style in formats:
            density, rank = zip(*points[fmt])
            ax.scatter(density, rank, color=color, label="%s (%s builds)" % (name, format(len(rank), ",")), **style)
        log_axes(ax)
        ax.set_title(title, pad=4)
        ax.set_xlabel("$N\\ell^{d}$ (points within one length scale)")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.set_ylim(*ylim)
        legend = ax.legend(loc="upper right", markerscale=1.8, handletextpad=0.3, borderaxespad=0.4,
                           framealpha=0.9, edgecolor="#cccccc")
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    axes[0].set_ylabel("factorization rank")
    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "rank_vs_density", png)


def main():
    args = [a for a in sys.argv[1:] if a != "--png"]
    out_dir = args[0] if args else "."
    png = "--png" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        figure_rank(out_dir, png)
        figure_h2_hodlr_gpu_2d(out_dir, png)
        figure_h2_hodlr_gpu_3d(out_dir, png)
        figure_scaling_sparse(out_dir, png)
        figure_history(out_dir, png)
        figure_h2_hodlr_cpu(out_dir, png)


if __name__ == "__main__":
    main()
