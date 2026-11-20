#! /usr/bin/env python
"""
The rank of the hierarchical matrix against how densely the sample resolves the kernel, for HODLR
(format 1) and H2 (format 7).

The x axis is N * l^d, the expected number of points in a box one length scale wide: the length
scale on its own carries the units of the domain and says nothing by itself, while N * l^d is
dimensionless, and it is what the rank of a block turns out to follow.  In 1D it is N * l.

Every build of a run (one H2 or HODLR matrix of K, constructed and factored at one hyperparameter value;
see k_builds) gives one point, and the plot is that scatter alone.  The worker log holds everything: the
kernel of every matrix it builds, then its ranks and accuracy.  Two ranks are available:

  factorisation (maxrank from d_py_bpack_factor)  both formats report it, so it is the one that
                                                  compares them
  construction  (rankmax_of_level)                only with --verbosity 1; the H2 path leaves the
                                                  init_compute counter at zero whatever the
                                                  verbosity

    python plot_rank_vs_density.py --objtype 1 [--kind factor|construct] [--title T]
                                   [--output F.pdf] <run dir> [<run dir> ...]

A run directory holds the worker_N*.log files of one format; which format it is comes from the options
the worker echoes, so the directory names do not matter.
"""
import argparse
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RC = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
      'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
      'figure.titlesize': 13}
FORMATS = {1: ('HODLR', 'tab:blue', 'o'), 7: ('H2', 'tab:red', 'X')}


def k_builds(worker_log, kind="factor"):
    """(format, [(N * l^d, rank, relative matrix-vector error)]) for every build in a worker log, in order.

    A build is one H2 or HODLR matrix of the covariance K, constructed and factored at the hyperparameters
    of one likelihood evaluation. The worker prints the george kernel of every matrix it builds, and a
    gradient evaluation also builds the derivative matrices dK/dtheta_j right after K; these are compressed
    but not factored, and are not builds. H2 checks K with 'H2_CheckError(quick)' and each derivative matrix
    with 'H2_CheckError(compression quick)'; HODLR factors only K. Each build is read whole from the stretch
    of the log between its kernel and the next, so its length scale, rank and error cannot be mismatched."""
    text = open(worker_log, errors="replace").read()
    fmt = 1 if re.search(r"'--format', '1'", text) else 7 if re.search(r"'--format', '7'", text) else None
    n = int(re.search(r"_N(\d+)", os.path.basename(worker_log)).group(1))
    out = []
    for seg in text.split("'kernel': ")[1:]:
        if fmt == 7:
            m = re.search(r"H2_CheckError\(quick\):[^\n]*?acc_mvp:\s*([0-9.eE+-]+)[\s\S]*?maxrank from d_py_bpack_factor:\s*(\d+)", seg)
            if not m:
                continue
            error, rank = float(m.group(1)), int(m.group(2))
        else:
            r = re.search(r"maxrank from d_py_bpack_factor:\s*(\d+)", seg)
            if not r:
                continue
            a = re.search(r"BPACK_CheckError\(mvp\):.*acc:\s*([0-9.eE+-]+)", seg)
            error, rank = (float(a.group(1)) if a else float("nan")), int(r.group(1))
        if kind == "construct":
            levels = [max(int(v) for v in m.group(1).split()) for m in re.finditer(r"rankmax_of_level:((?:\s+\d+)+)", seg)]
            rank = max(levels) if levels else 0
        # george's metric is l^2 per dimension, printed as one value with ndim (isotropic) or an array
        mm = re.search(r"metric=Metric\((?:array\(\[([^\]]*)\]\)|([0-9.eE+-]+), ndim=(\d+))", seg)
        if mm.group(1) is not None:
            metric = [float(v) for v in mm.group(1).split(",")]
        else:
            metric = [float(mm.group(2))] * int(mm.group(3))
        # N l^d with l the geometric mean of the length scales (what sets the volume of a correlated
        # neighbourhood); with the scaled-coordinate tree the worker sees the scaled metric instead
        out.append((n * math.exp(0.5 * sum(math.log(v) for v in metric)), rank, error))
    return fmt, out


def collect(directories, objtype, kind, keep=None):
    """{format: [(N * l^d, rank)]} over every build (see k_builds) of every run in the directories; keep(worker
    log path), if given, selects the runs."""
    points = defaultdict(list)
    for directory in directories:
        for worker_log in sorted(glob.glob(os.path.join(directory, "worker_N*.log"))):
            if keep is not None and not keep(worker_log):
                continue
            fmt, builds = k_builds(worker_log, kind)
            if fmt is not None:
                points[fmt] += [(density, rank) for density, rank, _ in builds if density > 0]
    return points


def main():
    parser = argparse.ArgumentParser(description="rank against N * l^d")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--objtype", type=int, required=True)
    parser.add_argument("--kind", choices=("factor", "construct"), default="factor")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    points = collect(args.directories, args.objtype, args.kind)
    if not points:
        raise SystemExit("no builds found under %s" % ", ".join(args.directories))
    what = "factorisation" if args.kind == "factor" else "construction"
    title = args.title or "%s rank against N l^d, objective %d" % (what.capitalize(), args.objtype)
    output = args.output or os.path.join(os.path.dirname(args.directories[0].rstrip("/")) or ".",
                                         "rank_vs_density_obj%d_%s.pdf" % (args.objtype, args.kind))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5.6))
        for fmt in sorted(points):
            name, color, marker = FORMATS[fmt]
            x = [d for d, _ in points[fmt]]
            y = [r for _, r in points[fmt]]
            ax.scatter(x, y, s=16, alpha=0.4, color=color, marker=marker, linewidths=0,
                       label="%s (%d builds)" % (name, len(x)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N\,\ell^{\,d}$   (points within one length scale)")
        ax.set_ylabel("%s rank" % what)
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(loc="best", markerscale=2.5)
        fig.tight_layout()
        for path in (output, os.path.splitext(output)[0] + ".png"):
            fig.savefig(path, dpi=150, bbox_inches="tight")
        print("wrote %s and %s" % (output, os.path.splitext(output)[0] + ".png"))
        plt.close(fig)

    for fmt in sorted(points):
        y = np.array([r for _, r in points[fmt]])
        print("%-6s %5d builds, rank min %3d median %3d max %3d"
              % (FORMATS[fmt][0], len(y), y.min(), int(np.median(y)), y.max()))


if __name__ == "__main__":
    main()
