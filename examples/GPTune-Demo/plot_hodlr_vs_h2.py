"""
HODLR (format 1) against H2 (format 7) over a range of sample counts, in four log-log panels:
model time, search time, model time per likelihood evaluation, and the number of likelihood
evaluations.  The optimizers are blue (gradient L-BFGS), red (finite-difference L-BFGS), purple
(MCMC) and green (MALA); the four format/device combinations have their own line style and marker.
An optimizer or combination that no run used is simply absent from the panels.

One figure holds one objective function: the run directories are named explicitly and a run of
another objective, or of an objective the figure was not asked for, is refused rather than drawn
beside it.

The directories are read in the order given and a later one wins, so a series can be assembled
from several campaigns: naming the unrestricted runs first and the restricted ones second takes
the restricted result wherever one exists and falls back on the unrestricted result elsewhere.
Only do that where the setting does not affect what is being measured, and say so in the title.

    python plot_hodlr_vs_h2.py --objtype 1 [--title T] [--output F.pdf] <run dir> [<run dir> ...]

Each run directory holds the run_stats_obj<objtype>_N<N>_<optimizer>.json files that
model_comparison_updated_bpack.py writes; a run belongs to HODLR or H2 by the "format" field of
those files, and to CPU or GPU by its directory name, so the names need only differ.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPTIMIZERS = [("gradient", "grad", "blue"),
              ("finite difference", "fd", "red"),
              ("mcmc", "mcmc", "purple"),
              ("mala", "mala", "green")]
SERIES = [(1, "gpu", "f1 hodlr gpu", "-", "o"),
          (7, "gpu", "f7 h2 gpu", "--", "X"),
          (1, "cpu", "f1 hodlr cpu", ":", "s"),
          (7, "cpu", "f7 h2 cpu", "-.", "^")]


def collect(directories, objtype):
    """{(format, device): {optimizer: {N: stats}}} over the named run directories."""
    runs = defaultdict(lambda: defaultdict(dict))
    found = 0
    for directory in directories:
        if not os.path.isdir(directory):
            raise SystemExit("not a directory: %s" % directory)
        # every objective present in the directory, so that a mixed one is refused outright
        others = {int(m.group(1)) for m in
                  (re.search(r"run_stats_obj(\d+)_N", os.path.basename(p))
                   for p in glob.glob(os.path.join(directory, "**", "run_stats_obj*_N*.json"), recursive=True))
                  if m} - {objtype}
        if others:
            raise SystemExit("%s also holds objective %s; one figure holds one objective function, "
                             "so give each its own run directories"
                             % (directory, ", ".join(str(o) for o in sorted(others))))
        for path in glob.glob(os.path.join(directory, "**", "run_stats_obj%d_N*_*.json" % objtype),
                              recursive=True):
            try:
                run = json.load(open(path))
            except (ValueError, OSError):
                print("skipping unreadable %s" % path)
                continue
            fmt = int(run.get("format", 0))
            if fmt not in (1, 7):
                continue
            device = "cpu" if "cpu" in os.path.dirname(path).rsplit("/", 1)[-1].lower() else "gpu"
            name = os.path.basename(path)
            n = int(re.search(r"_N(\d+)_", name).group(1))
            optimizer = run.get("optimizer",
                                re.search(r"_N\d+_(.+)\.json", name).group(1).replace("_", " "))
            stats = run["stats"]
            runs[(fmt, device)][optimizer][n] = {
                "time_model": stats["time_model"],
                "time_search": stats["time_search"],
                "time_per_eval": stats["time_model_per_likelihoodeval"],
                "iterations": stats["modeling_iteration"][0],
            }
            found += 1
    if not found:
        raise SystemExit("no run_stats_obj%d_N*_*.json files under %s" % (objtype, ", ".join(directories)))
    return runs


def legend_handles(runs):
    """One handle per optimizer (its colour) and one per format present (its style and marker)."""
    from matplotlib.lines import Line2D
    handles = []
    for optimizer, short, color in OPTIMIZERS:
        if any(runs.get(key, {}).get(optimizer) for key in runs):
            handles.append(Line2D([], [], color=color, linewidth=1.6, label=short))
    for fmt, device, fmt_label, style, marker in SERIES:
        if runs.get((fmt, device)):
            handles.append(Line2D([], [], color="0.35", linestyle=style, marker=marker,
                                  markersize=5, linewidth=1.6, label=fmt_label))
    return handles


def main():
    parser = argparse.ArgumentParser(description="HODLR against H2 over a range of sample counts")
    parser.add_argument("directories", nargs="+",
                        help="run directories, all of one objective; a later one overrides an earlier one")
    parser.add_argument("--objtype", type=int, required=True, help="the objective function to plot")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", default=None, help="default: hodlr_vs_h2_obj<objtype>.pdf beside the first directory")
    args = parser.parse_args()

    title = args.title or "HODLR (format 1) vs H2 (format 7), objective %d" % args.objtype
    output = args.output or os.path.join(os.path.dirname(args.directories[0].rstrip("/")) or ".",
                                         "hodlr_vs_h2_obj%d.pdf" % args.objtype)
    runs = collect(args.directories, args.objtype)

    fontsize = 8
    plt.rcParams.update({"font.size": fontsize})
    figure, axis = plt.subplots(2, 2, figsize=(11, 8.5))
    figure.suptitle(title, fontsize=fontsize + 2)
    panels = [(axis[0, 0], "time_model", "Model Time", "Time (sec)", "Sample Count"),
              (axis[0, 1], "time_search", "Search Time", "Time (sec)", "Sample Count"),
              (axis[1, 0], "time_per_eval", "Model Time Per Iteration", "Time (sec)", "Sample Count"),
              (axis[1, 1], "iterations", "Model Iterations", "Iterations", "Sample Count")]

    for ax, key, panel_title, ylabel, xlabel in panels:
        for fmt, device, fmt_label, style, marker in SERIES:
            for optimizer, short, color in OPTIMIZERS:
                series = runs.get((fmt, device), {}).get(optimizer, {})
                if not series:
                    continue
                counts = sorted(series)
                ax.loglog(counts, [series[n][key] for n in counts], style, color=color,
                          marker=marker, markersize=5, linewidth=1.6)
        ax.set_title(panel_title, fontsize=fontsize + 1)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        # One entry per optimizer and one per format, rather than one per combination of the two:
        # the colour carries the optimizer and the line style and marker carry the format, so a
        # combined legend repeats each colour once per format and reads as duplicated.
        ax.legend(handles=legend_handles(runs), fontsize=fontsize - 2, ncol=2)

    plt.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(os.path.splitext(output)[0] + ".png", dpi=150, bbox_inches="tight")
    print("wrote %s and %s" % (output, os.path.splitext(output)[0] + ".png"))

    for fmt, device, fmt_label, _, _ in SERIES:
        for optimizer, short, _ in OPTIMIZERS:
            series = runs.get((fmt, device), {}).get(optimizer, {})
            for n in sorted(series):
                s = series[n]
                print("%-13s %-18s N=%-8d model %9.1f s  search %7.1f s  %8.2f s/eval  %4d evals"
                      % (fmt_label, optimizer, n, s["time_model"], s["time_search"],
                         s["time_per_eval"], s["iterations"]))


if __name__ == "__main__":
    main()
