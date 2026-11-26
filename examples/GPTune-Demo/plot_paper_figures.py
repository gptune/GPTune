#!/usr/bin/env python3
"""
Figures 2-5 of the comparative-study paper, drawn at their printed size from the saved run files:

  rank_vs_density.pdf       factorization rank of every build against N l^d (plot_rank_vs_density.py reads
                            the ranks from the worker logs and the hyperparameters from the driver logs):
                            (a) objective 1 (oscillatory 1D) on CPUs, unrestricted length scale,
                            scal_obj1_h2cpu and scal_obj1_hodlrcpu; (b) objective 2 (2D isotropic) on GPUs,
                            length scale >= 0.01, scal_h2gpu_2d_iso and scal_hodlrgpu_2d_iso
  scaling_sparse.pdf        Wendland against INLA, 16 GPUs, in the layout of h2_vs_hodlr_scaling.pdf: time per
                            likelihood evaluation and training time against N, four optimizers, (a) the 2D Schwefel
                            objective (objective 4; Wendland scaling_obj4_reuse/, INLA inla_scaling_obj4/inla/) and
                            (b) the 3D isotropic quadratic (bpack objective 3, L-BFGS only; sparse_3d_obj3/)
  history_h2_hodlr_wendland_inla.pdf
                            Fig. history, training histories without markers: (a) -log likelihood, test RMSE and
                            the matrix-vector error of the build behind each history entry along the H2 and HODLR
                            trainings of the 2D anisotropic objective at N=200,000 (as history_200k_h2_hodlr.pdf);
                            (b) -log likelihood, test RMSE and test CRPS along the trainings of the Wendland GP at
                            N=100,000 (training_metrics files in this directory) and the INLA GP at N=102,400
                            (inla_vs_wendland/59033221/inla/); the same data as plot_history_with_inla.py
  h2_vs_hodlr_scaling.pdf   H2 against HODLR, time per likelihood evaluation and training time against N, four
                            optimizers: (a) objective 1 (oscillatory 1D) on CPUs, (b) objective 2 (2D
                            isotropic), (c) objective 4 (2D anisotropic) and (d) objective 5 (3D anisotropic)
                            on GPUs; the runs are listed in figure_h2_hodlr_scaling
  hmat_pattern.pdf          block structure of the HODLR and H2 matrices of K, compressed blocks colored by rank:
                            the points of objective 4 (2D), N=10,000, RBF with isotropic l = 0.1 (amplitude and
                            noise of the H2 gradient run at 100k), tol 1e-10, scaled tree; from the dumps of
                            hmat_pattern/hmat_pattern.sbatch (LENGTHSCALES=0.1)
  history_200k_h2_hodlr.pdf -log likelihood, test RMSE and the relative matrix-vector error of each entry's build along the
                            H2 and HODLR trainings of the 2D anisotropic objective (Fig. h2_vs_hodlr_scaling c) at
                            N=200,000, four optimizers (scal_{h2,hodlr}gpu_2d_aniso_scaled)
  history_1600k_obj1_h2_hodlr.pdf
                            the same for the 1D oscillatory objective on CPUs (Fig. h2_vs_hodlr_scaling a) at
                            N=1.6M, length scale in [0.02, 0.03] (scal_obj1ls_{h2,hodlr}cpu)

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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter

from plot_rank_vs_density import collect as rank_builds, k_builds

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


def run_stats(directories, objtype, fmt=None, recursive=True, required=True):
    """{optimizer: {N: stats}} from run_stats_obj<o>_N<N>_<optimizer>.json under the directories (or, with
    recursive=False, directly in them); a later directory overrides an earlier one for the same optimizer and N.
    Without any file: an error, or with required=False (runs still pending) no runs."""
    runs = {}
    pattern = os.path.join("**", "run_stats_obj%d_N*_*.json" % objtype) if recursive else "run_stats_obj%d_N*_*.json" % objtype
    paths = [sorted(glob.glob(os.path.join(d, pattern), recursive=recursive)) for d in directories]
    if not any(paths) and required:
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


SPARSE_3D = os.path.join(HERE, "sparse_3d_obj3")
SPARSE_3D_N = (100000, 400000, 800000)
# the 3D Wendland runs: sparse_3d_obj3 (zero prior mean, jobs 61771465/66/91) or sparse_3d_obj3_mean (prior mean set
# to the sample mean of the training outputs, GP_MEAN=sample, as the INLA model; jobs 61923410/12)
WENDLAND_3D = os.path.join(HERE, os.environ.get("WENDLAND_3D", "sparse_3d_obj3"))


def model_handles():
    """legend entries of the two sparse models: Wendland dash-dotted with hollow markers, INLA solid with filled
    ones"""
    return [Line2D([], [], color="#555555", marker="o", markersize=3, markerfacecolor="white", linestyle="-.",
                   label="Wendland"),
            Line2D([], [], color="#555555", marker="o", markersize=3, linestyle="-", label="INLA")]


def figure_scaling_sparse(out_dir, png):
    """Wendland against INLA (Fig. scaling_sparse), in the layout of Fig. h2_vs_hodlr_scaling: time per likelihood
    evaluation (top) and training time (bottom) against N for the four optimizers, 16 GPUs, one column per objective.
      (a) objective 4 (2D Schwefel): Wendland scaling_obj4_reuse, INLA inla_scaling_obj4/inla. The Wendland MCMC and
          MALA runs are the capped runs of 27 Sep (as in Table III); the uncapped MALA reruns in
          scaling_obj4_mala_uncapped have no run_stats (the 100k one was stopped by its wall time, see its
          timeout_6h_61634303/step700_summary.txt) and its subdirectories hold superseded attempts, so it is not read
      (b) objective 3 of the bpack set (3D isotropic quadratic), gradient and FD L-BFGS only: sparse_3d_obj3/
          wendland_N<N> and inla_nu0.5_N<N> (jobs 61771465/66/91, run_sparse_steps_gpu.sbatch); until those INLA runs
          are in, the earlier gradient-only INLA runs inla_3d/obj3_N<N>_nu0.5 stand in for them (a run of
          sparse_3d_obj3 takes precedence for the same optimizer and N)"""
    inla_3d = run_stats([os.path.join(HERE, "inla_3d", "obj3_N%d_nu0.5" % n) for n in SPARSE_3D_N]
                        + [os.path.join(SPARSE_3D, "inla_nu0.5_N%d" % n) for n in SPARSE_3D_N], 3)
    columns = [("(a) 2D Schwefel",
                run_stats([os.path.join(HERE, "scaling_obj4_reuse")], 4),
                run_stats([os.path.join(HERE, "inla_scaling_obj4", "inla")], 4),
                [1e5, 2e5, 4e5, 8e5, 1.6e6], ["100k", "200k", "400k", "800k", "1.6M"], (8.5e4, 1.9e6)),
               ("(b) 3D isotropic quadratic",
                run_stats([os.path.join(WENDLAND_3D, "wendland_N%d" % n) for n in SPARSE_3D_N], 3, required=False),
                inla_3d,
                [1e5, 2e5, 4e5, 8e5], ["100k", "200k", "400k", "800k"], (8.5e4, 9.4e5))]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="col", sharey="row")
    for col, (title, wendland, inla, ticks, labels, xlim) in enumerate(columns):
        for name, runs in (("Wendland", wendland), ("INLA", inla)):
            print("%s, %s:" % (title, name), {o: sorted(s) for o, s in sorted(runs.items())} or "no runs yet")
        for row, key in enumerate(("time_per_eval", "time_model")):
            ax = axes[row, col]
            draw_series(ax, wendland, key, hollow=True, linestyle="-.")
            draw_series(ax, inla, key)
            log_axes(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(*xlim)
        axes[0, col].set_title(title, pad=4)
        axes[1, col].set_xlabel("training points $N$")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    axes[0, 0].set_ylabel("time per evaluation (s)")
    axes[1, 0].set_ylabel("training time (s)")
    fig.tight_layout(h_pad=0.5, w_pad=1.0, rect=(0, 0, 1, 0.93))
    fig.legend(handles=optimizer_handles() + model_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, columnspacing=1.6, handlelength=3.2)
    save(fig, out_dir, "scaling_sparse", png)


def training_metrics(directory, N, optimizer, objtype=4):
    path = os.path.join(directory, "training_metrics_obj%d_N%d_%s.csv" % (objtype, N, optimizer.replace(" ", "_")))
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


HISTORY_XLIM = (7e-3, 6e3)


def optimizer_lines():
    """legend entries of the four optimizers as plain lines (the history curves have no markers)"""
    return [Line2D([], [], color=STYLE[o][1], linewidth=1.0, label=STYLE[o][0]) for o in OPTIMIZERS]


def history_axes(axes, ylabels):
    """log time axes of one row of training histories; log vertical axes except the first (-log likelihood)"""
    for ax, ylabel in zip(axes, ylabels):
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", **GRID)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    for ax in axes[1:]:
        ax.set_yscale("log")


def draw_history_wendland_inla(axes):
    """-log likelihood, test RMSE and test CRPS against the training time, Wendland and INLA at ~100k, into three axes,
    curves without markers (Wendland solid, INLA dash-dotted and a little thicker, so that its flat curves at the
    bottom of the panels stay visible); returns the legend entries of the two line styles.
    Both MALA curves are uncapped runs with the default stagnation stop: Wendland = the 100k run of Table III
    (tab:wend_opt, job 61694865), INLA = the MALA row of Table IV (tab:schwefel_opt, job 61707211), 40 replayed
    history entries each. The Wendland MCMC curve is the uncapped run with the stagnation stop and the chain-agreement
    test (row "MCMC, no cap" of Table III, job 61669899, scaling_obj4_stagnation); it ends at 15,856 s, beyond the
    6,000 s axis. The other curves are the runs in HERE and inla_vs_wendland/59033221/inla."""
    models = [("Wendland", HERE, 100000, ("-", 1.0),
               {"mala": os.path.join(HERE, "scaling_obj4_mala_stagnation"),
                "mcmc": os.path.join(HERE, "scaling_obj4_stagnation")}),
              ("INLA", os.path.join(HERE, "inla_vs_wendland", "59033221", "inla"), 102400, ("-.", 1.4),
               {"mala": os.path.join(HERE, "inla_vs_wendland", "mala_stagnation", "inla")})]
    for label, directory, N, (linestyle, width), run_dirs in models:
        for optimizer in OPTIMIZERS:
            rows = training_metrics(run_dirs.get(optimizer, directory), N, optimizer)
            sampler = optimizer in ("mcmc", "mala")
            style = dict(line_style(optimizer, linestyle=linestyle), marker="none", linewidth=width)
            for ax, key in zip(axes, ("nll", "rmse", "crps")):
                keep = np.isfinite(rows[key]) & (rows["time"] > 0)
                assert keep.sum() == np.isfinite(rows[key]).sum(), "an entry at time 0 would vanish on the log axis"
                ax.plot(rows["time"][keep], rows[key][keep], drawstyle="steps-post" if sampler else "default", **style)
    history_axes(axes, ("negative log-likelihood", "test RMSE (1,000 points)", "test CRPS (1,000 points)"))
    for ax in axes[1:]:
        ax.set_ylim(8e-5, 1.2)
    # direct labels for the two models, as well as the line styles in the legend
    text = dict(fontsize=7.5, color="#333333", ha="left")
    axes[0].set_ylim(-6.4e5, 1.9e5)
    axes[0].text(0.012, 1.2e5, "Wendland", va="bottom", **text)
    axes[0].text(0.012, -4.2e5, "INLA", va="bottom", **text)
    # INLA's RMSE stays flat near 1.4e-4 (label above it), its CRPS falls from 4.8e-4 (label below)
    for ax, inla_y in zip(axes[1:], (2.0e-4, 1.0e-4)):
        ax.text(0.012, 0.32, "Wendland", va="bottom", **text)
        ax.text(0.012, inla_y, "INLA", va="bottom", **text)
    return [Line2D([], [], color="#555555", linewidth=width, linestyle=ls, label=l) for l, _, _, (ls, width), _ in models]


def figure_history(out_dir, png):
    """Fig. history, two rows of training histories against the training time, curves without markers:
    (a) H2 (solid) and HODLR (dashed) on the 2D anisotropic objective at N=200,000 on GPUs, the runs of
        Fig. h2_vs_hodlr_scaling (c) (draw_history_h2_hodlr): -log likelihood, test RMSE, error of each build;
    (b) Wendland (solid) and INLA (dash-dotted) on the 2D Schwefel objective at ~100k (draw_history_wendland_inla):
        -log likelihood, test RMSE, test CRPS.
    The columns share the time axis; the -log likelihoods are in units of 1e5."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    rows = [("(a) $\\mathcal{H}^2$ and HODLR, 2D anisotropic objective, $N$ = 200,000", draw_history_h2_hodlr(axes[0])),
            ("(b) Wendland ($N$ = 100,000) and INLA ($N$ = 102,400), 2D Schwefel objective",
             draw_history_wendland_inla(axes[1]))]
    for ax in axes.flat:
        ax.set_xlim(*HISTORY_XLIM)
    for ax in axes[1]:
        ax.set_xlabel("training time (s)")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % (v / 1e5)).replace("-", "\u2212")))
        ax.set_ylabel("$-$log-likelihood ($\\times 10^5$)")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.875, wspace=0.42, hspace=0.48)
    fig.legend(handles=optimizer_lines(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               columnspacing=1.6, handlelength=2.4)
    # each row: its title on the left, the line styles of its two models on the right
    for r, (title, keys) in enumerate(rows):
        y = axes[r, 0].get_position().y1 + 0.03
        fig.text(0.012, y, title, ha="left", va="center", fontsize=8)
        fig.legend(handles=keys, loc="center right", bbox_to_anchor=(axes[r, 2].get_position().x1, y),
                   ncol=2, frameon=False, handlelength=2.4, columnspacing=1.2, borderaxespad=0, borderpad=0)
    save(fig, out_dir, "history_h2_hodlr_wendland_inla", png)


TRAINING_RECORD = re.compile(r"(?:L-BFGS iteration|MCMC best sample) \d+: time ([0-9.eE+-]+) s")
TIME_SPENT = re.compile(r"Time spent in ([^:]*): ([0-9.eE+-]+) seconds")


def build_times(driver_log, time_model):
    """(training time of each build made during the training, number of builds in the whole log) of a GPTune
    run with ButterflyPACK. The driver prints one 'compress and invert K' line per build, in the order of the
    worker's builds (plot_rank_vs_density.k_builds), and records the training time at every L-BFGS iteration or
    MCMC improvement; the replays of the test metrics come after the training. A build's time is interpolated
    between those records in the compression work the driver logs ('Time spent in ...'), with the model time
    of the run as the last record. Each replay prints its test-metrics line after its build, so the build
    before the first such line is the first replay, not a training build."""
    work, builds, records, total, training, last = 0.0, [], [], 0, True, 0.0
    with open(driver_log, errors="replace") as fh:
        for line in fh:
            if "test metrics at history entry" in line and training:
                training = False
                builds.pop()
                work -= last
            m = TIME_SPENT.search(line)
            if m:
                if m.group(1) == "compress and invert K":
                    total += 1
                if training:
                    work += float(m.group(2))
                    if m.group(1) == "compress and invert K":
                        builds.append(work)
                        last = float(m.group(2))
                continue
            m = TRAINING_RECORD.search(line)
            if m and training:
                records.append((work, float(m.group(1))))
    records.append((work, time_model))
    w, t = np.array(records).T
    t = np.maximum.accumulate(t)
    keep = np.append(w[1:] > w[:-1], True)   # one record per amount of work: the last one
    return np.interp(builds, w[keep], t[keep]), total


def entry_build_errors(rows, builds):
    """The relative error of the build behind each history entry (training_metrics rows of a george RBF model:
    hp_0 log noise variance, hp_1 log constant, hp_2... log metric), from the training builds of
    k_builds(..., kernel=True): the build closest to the entry's hyperparameters (in the logs of the noise
    variance, constant and metric) at or after the build of the previous entry, so an entry repeated at the end
    of the training keeps its build. The worker prints the metric with as few as 5 significant digits, so the
    match is the nearest build within 1e-4 (finite-difference steps around an iterate are larger: scipy's
    3-point step is 6e-6 relative); NaN if none is that close."""
    hps = sorted(k for k in rows if k.startswith("hp_"))
    params = np.array([[2 * np.log(b[3][2]), b[3][0]] + list(np.log(b[3][1])) for b in builds])
    errors, start = np.full(len(rows["time"]), np.nan), 0
    for i in range(len(errors)):
        hp = np.array([rows[k][i] for k in hps])
        distance = np.abs(params[start:, :len(hp)] - hp).max(axis=1)
        j = int(np.argmin(distance))
        if distance[j] < 1e-4:
            errors[i], start = builds[start + j][2], start + j
    return errors


def draw_history_h2_hodlr(axes, objtype=4, N=200000, h2_dir="scal_h2gpu_2d_aniso_scaled",
                          hodlr_dir="scal_hodlrgpu_2d_aniso_scaled"):
    """Draws -log likelihood, test RMSE and the relative error of each build against the training time, H2 and HODLR,
    into three axes (returns the legend entries of the two line styles),
    the four optimizers, from one pair of runs of Fig. h2_vs_hodlr_scaling (run directories under CMP):
      history_200k_h2_hodlr        (c) 2D anisotropic (objective 4) at N=200,000 on GPUs, scal_h2gpu_2d_aniso_scaled
                                   (job 61605207) and scal_hodlrgpu_2d_aniso_scaled (job 61608561), scaled tree,
                                   tolerance 1e-10 for both
      history_1600k_obj1_h2_hodlr  (a) 1D oscillatory (objective 1) at N=1.6M on CPUs, length scale in [0.02, 0.03]:
                                   scal_obj1ls_h2cpu (job 59504958, tolerance 1e-11, h2_id_proxy 0) and
                                   scal_obj1ls_hodlrcpu (job 59504955, tolerance 1e-10)
    The third panel is the matrix-vector check of Table II (tab:hacc) of the build behind each history entry of the
    other two (entry_build_errors; the RMSE comes from a separate replay build at the same hyperparameters), as
    steps, with each format's tolerance (from its worker log) as a dotted line. All curves without markers: H2
    solid, HODLR dashed. MCMC and MALA show the best sample so far and stopped at their cap, 3x the shorter
    L-BFGS training."""
    formats = [("$\\mathcal{H}^2$", os.path.join(CMP, h2_dir), False, "-"),
               ("HODLR", os.path.join(CMP, hodlr_dir), True, "--")]
    log_tag = {"gradient": "grad", "finite difference": "fd", "mcmc": "mcmc", "mala": "mala"}
    tolerances = {}
    for label, directory, hollow, linestyle in formats:
        for optimizer in OPTIMIZERS:
            rows = training_metrics(directory, N, optimizer, objtype)
            style = line_style(optimizer, hollow=hollow, linestyle=linestyle)
            style.update(markersize=2.6, linewidth=1.0)
            stats = json.load(open(os.path.join(directory, "run_stats_obj%d_N%d_%s.json"
                                                % (objtype, N, optimizer.replace(" ", "_")))))
            times, total = build_times(os.path.join(directory, "driver_N%d_%s.log" % (N, log_tag[optimizer])),
                                       first(stats["stats"]["time_model"]))
            worker_log = os.path.join(directory, "worker_N%d_%s.log" % (N, log_tag[optimizer]))
            builds = k_builds(worker_log, kernel=True)[1]
            assert len(builds) == total, (directory, optimizer, len(builds), total)   # one worker build per driver build
            tolerance = re.search(r"'--tol_comp', '([^']+)'", open(worker_log, errors="replace").read())
            tolerances.setdefault(float(tolerance.group(1)), set()).add(label)
            # the error of the build behind each history entry of the other two panels
            rows = dict((key, rows[key]) for key in rows.dtype.names)
            rows["error"] = entry_build_errors(rows, builds[:len(times)])
            # curves without markers, as in Fig. history: the line style alone tells the formats apart
            # (the build errors as steps for every optimizer: an iterate's error holds until the next one)
            for ax, key in zip(axes, ("nll", "rmse", "error")):
                keep = np.isfinite(rows[key]) & (rows["time"] > 0)
                assert keep.sum() == np.isfinite(rows[key]).sum(), "an entry at time 0 would vanish on the log axis"
                ax.plot(rows["time"][keep], rows[key][keep],
                        drawstyle="steps-post" if optimizer in ("mcmc", "mala") or key == "error" else "default",
                        **dict(style, marker="none"))
            matched = np.isfinite(rows["error"])
            print("%s %-17s %3d of %3d history entries matched to training builds, error median %.1e max %.1e"
                  % (label, optimizer, matched.sum(), len(matched), np.median(rows["error"][matched]),
                     rows["error"][matched].max()))
    history_axes(axes, ("negative log-likelihood", "test RMSE (1,000 points)", "relative error of each build"))
    # one tolerance: label below its line; one per format: each label above its line (the lower one between the lines)
    for tolerance, labels in sorted(tolerances.items()):
        exponent = int(round(np.log10(tolerance)))
        if len(tolerances) == 1:
            text, y, va = "tolerance $10^{%d}$" % exponent, 0.8 * tolerance, "top"
        else:
            text = "%s tolerance $10^{%d}$" % (" and ".join(l for l, *_ in formats if l in labels), exponent)
            y, va = 1.25 * tolerance, "bottom"
        axes[2].axhline(tolerance, color="#777777", linewidth=0.6, linestyle=":", zorder=1)
        axes[2].text(0.03, y, text, transform=axes[2].get_yaxis_transform(), ha="left",
                     va=va, fontsize=6.5, color="#555555")
    return [Line2D([], [], color="#555555", linewidth=1.0, linestyle=ls, label=l) for l, _, _, ls in formats]


def figure_history_h2_hodlr(out_dir, png, objtype=4, N=200000, h2_dir="scal_h2gpu_2d_aniso_scaled",
                            hodlr_dir="scal_hodlrgpu_2d_aniso_scaled", name="history_200k_h2_hodlr"):
    """One row of draw_history_h2_hodlr as a figure of its own (history_200k_h2_hodlr, history_1600k_obj1_h2_hodlr)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    keys = draw_history_h2_hodlr(axes, objtype, N, h2_dir, hodlr_dir)
    for ax in axes:
        ax.set_xlabel("training time (s)")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axes[0].yaxis.set_major_formatter(formatter)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.legend(handles=optimizer_lines() + keys, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, columnspacing=1.4, handlelength=2.4)
    save(fig, out_dir, name, png)


def figure_h2_hodlr_scaling(out_dir, png):
    """H2 against HODLR (Fig. 5): time per likelihood evaluation (top) and training time (bottom: the whole
    modeling phase, time per evaluation x evaluations) against N for the four optimizers, one column per case.
      (a) objective 1 (oscillatory 1D) on CPUs: H2 scal_obj1_h2cpu (unrestricted length scale) with
          scal_obj1ls_h2cpu (length scale in [0.02, 0.03]) taking precedence where both have the run, i.e. from
          1.6M on; HODLR scal_obj1ls_hodlrcpu (restricted throughout)
      (b) objective 2 (2D isotropic) on GPUs: H2 scal_h2gpu_2d_iso to 1.6M, HODLR scal_hodlrgpu_2d_iso to 200k
      (c) objective 4 (2D anisotropic) on GPUs, scaled-coordinate tree: H2 scal_h2gpu_2d_aniso_scaled at 100k,
          200k and 1.6M, and the earlier raw-coordinate runs scal_h2gpu_2d_aniso at 400k and 800k (dagger);
          HODLR scal_hodlrgpu_2d_aniso_scaled to 200k
      (d) objective 5 (3D anisotropic) on GPUs: H2 scal_h2gpu_3d_aniso_scaled at 50k, 100k and 400k, and the
          raw-coordinate scal_h2gpu_3d_aniso at 200k (dagger); HODLR scal_hodlrgpu_3d_aniso_scaled at 25k
    The GPU runs are the rows of the former Table IX (tab:h2gpu) and the larger H2 runs: ButterflyPACK 3f05cd5
    (iso) and eef6ed0/eb5e78b (scaled) to 800k, 1d3d6d1 at 1.6M and for the 3D 50k and 400k runs
    (submit_h2_large.sh, submit_aniso_scaled.sh; no MALA). Every MCMC and MALA run stopped at its time cap,
    3x the shorter L-BFGS training."""
    columns = [("(a) 1D oscillatory, CPU",
                run_stats([os.path.join(CMP, "scal_obj1_h2cpu"), os.path.join(CMP, "scal_obj1ls_h2cpu")], 1, fmt=7),
                run_stats([os.path.join(CMP, "scal_obj1ls_hodlrcpu")], 1, fmt=1),
                [1e5, 4e5, 1.6e6, 6.4e6, 2.56e7], ["100k", "400k", "1.6M", "6.4M", "25.6M"], (8e4, 3.2e7)),
               ("(b) 2D isotropic, GPU",
                table_runs([("scal_h2gpu_2d_iso", [100000, 200000, 400000, 800000, 1600000])], 2, 7),
                # (HODLR 200k MCMC sampled for 0.5 s, 1 step: its cap, 3x the shorter L-BFGS training, came from
                #  the 0.17 s of the failed gradient run, so its training time measures nothing; its 4 evaluations
                #  still give its time per evaluation)
                table_runs([("scal_hodlrgpu_2d_iso", [100000, 200000])], 2, 1),
                [1e5, 2e5, 4e5, 8e5, 1.6e6], ["100k", "200k", "400k", "800k", "1.6M"], (8.5e4, 1.9e6)),
               ("(c) 2D anisotropic, GPU",
                # (1.6M without gradient L-BFGS: its derivative compression for the long length scale took
                #  151 s per evaluation (median) against 1.4 s at 200k, a ButterflyPACK slowdown, 20 Nov 2026)
                table_runs([("scal_h2gpu_2d_aniso_scaled", [100000, 200000]),
                            ("scal_h2gpu_2d_aniso_scaled", [1600000], ("gradient",)),
                            ("scal_h2gpu_2d_aniso", [400000, 800000])], 4, 7),
                table_runs([("scal_hodlrgpu_2d_aniso_scaled", [100000, 200000])], 4, 1),
                [1e5, 2e5, 4e5, 8e5, 1.6e6], ["100k", "200k", "400k\u2020", "800k\u2020", "1.6M"], (8.5e4, 1.9e6)),
               ("(d) 3D anisotropic, GPU",
                table_runs([("scal_h2gpu_3d_aniso_scaled", [50000, 100000, 400000]),
                            ("scal_h2gpu_3d_aniso", [200000])], 5, 7),
                table_runs([("scal_hodlrgpu_3d_aniso_scaled", [25000])], 5, 1),
                [2.5e4, 5e4, 1e5, 2e5, 4e5], ["25k", "50k", "100k", "200k\u2020", "400k"], (2.1e4, 4.7e5))]
    broken = {"(b) 2D isotropic, GPU": {("mcmc", 200000)}}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex="col", sharey="row")
    for col, (title, h2, hodlr, ticks, labels, xlim) in enumerate(columns):
        for row, key in enumerate(("time_per_eval", "time_model")):
            ax = axes[row, col]
            draw_series(ax, without(hodlr, broken.get(title, set())) if key == "time_model" else hodlr, key,
                        hollow=True, linestyle="--")
            draw_series(ax, h2, key)
            log_axes(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(*xlim)
        axes[0, col].set_title(title, pad=4)
        axes[1, col].set_xlabel("training points $N$")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    axes[0, 0].set_ylabel("time per evaluation (s)")
    axes[1, 0].set_ylabel("training time (s)")
    fig.tight_layout(h_pad=0.5, w_pad=0.6, rect=(0, 0, 1, 0.93))
    fig.legend(handles=optimizer_handles() + format_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, columnspacing=1.6)
    save(fig, out_dir, "h2_vs_hodlr_scaling", png)


def format_handles():
    """legend entries of the two formats: H2 solid with filled markers, HODLR dashed with hollow ones"""
    return [Line2D([], [], color="#555555", marker="o", markersize=3, linestyle="-", label="$\\mathcal{H}^2$"),
            Line2D([], [], color="#555555", marker="o", markersize=3, markerfacecolor="white", linestyle="--",
                   label="HODLR")]


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


def without(runs, points):
    """runs without the (optimizer, N) points"""
    return {o: {n: stats for n, stats in series.items() if (o, n) not in points} for o, series in runs.items()}


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


# the paper figure: isotropic l = 0.1 (N10000_l0.1); N10000 holds the fitted anisotropic l = (0.112, 0.986), where
# the scaled points form an 8.9 x 1.0 strip and the two formats have similar ranks
PATTERN_DIR = os.path.join(HERE, "hmat_pattern", "N10000_l0.1")
DENSE_COLOR = "#c0504d"   # inadmissible (dense) blocks, outside the blue rank scale


def read_dumps(pattern):
    """lines of the BPACK_STRUCTURE_DUMP files of one build (one file per MPI rank)"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    lines = []
    for f in files:
        with open(f) as fh:
            lines += [l.split() for l in fh if l.strip() and not l.startswith("#")]
    return lines


def hodlr_blocks(directory):
    """[(row0, nrows, col0, ncols, rank or None for dense)] of the HODLR matrix, 0-based in its tree order"""
    blocks = []
    for v in read_dumps(os.path.join(directory, "blocks_hodlr_construct_0_r*.txt")):
        level_c, ii, headm, m, headn, n, rankmax, style, lb, ncols_u = map(int, v)
        if style == 1:
            blocks.append((headm - 1, m, headn - 1, n, None))
        else:
            blocks.append((headm - 1, m, headn - 1, n, ncols_u if ncols_u >= 0 else rankmax))
    return blocks


def h2_blocks(directory):
    """[(row0, nrows, col0, ncols, rank or None for dense)] of the H2 matrix in the Morton order of its leaf
    boxes; an admissible block's rank is min(k_t, k_s), the smaller skeleton of its two boxes (its coupling
    matrix K(S_t, S_s) is k_t x k_s)"""
    lines = read_dumps(os.path.join(directory, "blocks_h2_compress_0_r*.txt"))
    leaves = {}
    for v in lines:
        if v[0] == "P":
            leaves[(int(v[1]), int(v[2]))] = int(v[3])
    depth = {lv for lv, _ in leaves}
    assert len(depth) == 1, depth
    depth = depth.pop()
    morton = np.array(sorted(m for _, m in leaves))
    offsets = np.concatenate([[0], np.cumsum([leaves[(depth, m)] for m in morton])])

    def box_range(level, m):
        span = 4 ** (depth - level)
        a, b = np.searchsorted(morton, [m * span, (m + 1) * span])
        return offsets[a], offsets[b] - offsets[a]

    blocks = []
    for v in lines:
        if v[0] in "FN":
            level, t, s, rows, cols = map(int, v[1:])
            r0, nr = box_range(level, t)
            c0, nc = box_range(level, s)
            blocks.append((r0, nr, c0, nc, None if v[0] == "N" else min(rows, cols)))
    return blocks


def occupancy(n, rows, cols, bins):
    """bins x bins boolean image of an n x n sparsity pattern: a pixel is set if any entry in it is nonzero"""
    image = np.zeros((bins, bins), dtype=bool)
    image[np.asarray(rows, dtype=np.int64) * bins // n, np.asarray(cols, dtype=np.int64) * bins // n] = True
    return image


def figure_hmat_pattern(out_dir, png, directory=PATTERN_DIR, name="hmat_pattern"):
    """The four matrices on the same points (Fig. 1): the HODLR (a) and H2 (b) matrices of K, compressed blocks
    colored by rank on one shared scale (logarithmic when the ranks span more than a factor of 10), dense blocks
    in DENSE_COLOR, from the dumps of hmat_pattern/hmat_pattern.sbatch (same points, hyperparameters, tolerance
    and scaled coordinates); the Wendland covariance (c, in the order of the H2 tree) and the INLA posterior
    precision Q_c (d, lattice order) from hmat_pattern/sparse_patterns.py, nonzeros in DENSE_COLOR and zeros
    white (rank 0 on the color scale)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    from matplotlib.patches import Patch, Rectangle

    panels = [("(a) HODLR $K$", hodlr_blocks(directory)), ("(b) $\\mathcal{H}^2$ $K$", h2_blocks(directory))]
    n = int(np.sqrt(sum(nr * nc for _, nr, _, nc, _ in panels[0][1])))
    for title, blocks in panels:
        area = sum(nr * nc for _, nr, _, nc, _ in blocks)
        assert area == n * n, (title, area, n * n)   # the blocks tile the matrix
    all_ranks = [r for _, blocks in panels for *_, r in blocks if r is not None]
    rmin, rmax = max(min(all_ranks), 1), max(all_ranks)
    if rmax > 10 * rmin:
        norm = LogNorm(vmin=10 ** np.floor(np.log10(rmin)), vmax=10 ** np.ceil(np.log10(rmax)))
    else:
        norm = Normalize(vmin=0, vmax=10 * np.ceil(rmax / 10))
    cmap = plt.get_cmap("Blues")
    sparse = np.load(os.path.join(directory, "sparse_patterns.npz"))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25))
    for ax, (title, blocks) in zip(axes, panels):
        rects, colors = [], []
        for r0, nr, c0, nc, rank in blocks:
            rects.append(Rectangle((c0, r0), nc, nr))
            colors.append(DENSE_COLOR if rank is None else cmap(norm(max(rank, norm.vmin))))
        ax.add_collection(PatchCollection(rects, facecolors=colors, edgecolors="#303030", linewidths=0.08))
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ranks = [r for *_, r in blocks if r is not None]
        ndense = sum(r is None for *_, r in blocks)
        ax.set_title(title)
        ax.set_xlabel("%s compressed blocks\n(rank %d\u2013%d), %d dense"
                      % (format(len(ranks), ","), min(ranks), max(ranks), ndense), fontsize=6.5)
        print(title, "admissible", len(ranks), "ranks", min(ranks), int(np.median(ranks)), max(ranks),
              "dense", ndense)
    nonzero_cmap = ListedColormap(["white", DENSE_COLOR])
    for ax, key, title in [(axes[2], "wendland", "(c) Wendland $K$"), (axes[3], "inla", "(d) INLA $Q_c$")]:
        size = int(sparse[key + "_n"])
        nnz = len(sparse[key + "_i"])
        # about the printed resolution (a 1.5-inch panel at 300 dpi), so that every pixel with a nonzero shows
        image = occupancy(size, sparse[key + "_i"], sparse[key + "_j"], min(size, 450))
        ax.imshow(image, cmap=nonzero_cmap, vmin=0, vmax=1, extent=(0, size, size, 0), interpolation="none",
                  aspect="equal")
        ax.set_title(title)
        if key == "wendland":
            label = "$r_c=%.2f$, %.0f nonzeros\nper row (%.2f%%)" % (
                float(sparse["wendland_rc"]), nnz / size, 100.0 * nnz / size ** 2)
        else:
            m = sparse["inla_shape"]
            label = "%d\u00d7%d nodes, %.0f nonzeros\nper row (%.2f%%)" % (
                m[0], m[1], nnz / size, 100.0 * nnz / size ** 2)
        ax.set_xlabel(label, fontsize=6.5)
        print(title, size, nnz, nnz / size)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 0.915, 1))
    cax = fig.add_axes([0.928, 0.4, 0.013, 0.45])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("rank of compressed block", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)
    fig.legend(handles=[Patch(facecolor=DENSE_COLOR, edgecolor="#303030", linewidth=0.3, label="dense or\nnonzero"),
                        Patch(facecolor="white", edgecolor="#303030", linewidth=0.3, label="zero")],
               loc="lower left", bbox_to_anchor=(0.918, 0.02), frameon=False, handlelength=1.0, handletextpad=0.4,
               fontsize=6.5)
    save(fig, out_dir, name, png)


def main():
    args = [a for a in sys.argv[1:] if a != "--png"]
    out_dir = args[0] if args else "."
    png = "--png" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        figure_rank(out_dir, png)
        figure_scaling_sparse(out_dir, png)
        figure_history(out_dir, png)
        figure_history_h2_hodlr(out_dir, png)
        figure_history_h2_hodlr(out_dir, png, objtype=1, N=1600000, h2_dir="scal_obj1ls_h2cpu",
                                hodlr_dir="scal_obj1ls_hodlrcpu", name="history_1600k_obj1_h2_hodlr")
        figure_h2_hodlr_scaling(out_dir, png)
        figure_hmat_pattern(out_dir, png)


if __name__ == "__main__":
    main()
