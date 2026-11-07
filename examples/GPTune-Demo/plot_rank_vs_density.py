#! /usr/bin/env python
"""
The rank of the hierarchical matrix against how densely the sample resolves the kernel, for HODLR
(format 1) and H2 (format 7).

The x axis is N * l^d, the expected number of points in a box one length scale wide: the length
scale on its own carries the units of the domain and says nothing by itself, while N * l^d is
dimensionless, and it is what the rank of a block turns out to follow.  In 1D it is N * l.

Every build of a run gives one point, and the plot is that scatter alone.  The driver log records the george hyperparameter vector
before each build, the worker log the ranks of the same builds in the same order, so the two are
read together.  Two ranks are available:

  factorisation (maxrank from d_py_bpack_factor)  both formats report it, so it is the one that
                                                  compares them
  construction  (rankmax_of_level)                only with --verbosity 1; the H2 path leaves the
                                                  init_compute counter at zero whatever the
                                                  verbosity

    python plot_rank_vs_density.py --objtype 1 [--kind factor|construct] [--title T]
                                   [--output F.pdf] <run dir> [<run dir> ...]

A run directory holds the driver_N*.log and worker_N*.log pairs of one format; which format it is
comes from the options the worker echoes, so the directory names do not matter.
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


def builds_of(driver_log):
    """[(hyperparameters, seconds)] in the order the run built them."""
    out, pending = [], None
    for line in open(driver_log, errors="replace"):
        m = re.match(r"\[(.*?)\] george hyperparameters", line)
        if m:
            pending = [float(v) for v in m.group(1).split()]
        m = re.match(r"Time spent in compress and invert K: ([0-9.]+)", line)
        if m and pending is not None:
            out.append((pending, float(m.group(1))))
            pending = None
    return out


def ranks_of(worker_log, kind):
    """[rank] for every matrix the worker reported, in order, and the format it was built with."""
    text = open(worker_log, errors="replace").read()
    fmt = 1 if re.search(r"'--format', '1'", text) else 7 if re.search(r"'--format', '7'", text) else None
    if kind == "factor":
        ranks = [int(m.group(1)) for m in re.finditer(r"maxrank from d_py_bpack_factor:\s*(\d+)", text)]
    else:
        ranks = [max(int(v) for v in m.group(1).split())
                 for m in re.finditer(r"rankmax_of_level:((?:\s+\d+)+)", text)]
    return ranks, fmt


def collect(directories, objtype, kind):
    """{format: [(N * l^d, rank)]} over every build of every run in the directories."""
    points = defaultdict(list)
    for directory in directories:
        for driver_log in sorted(glob.glob(os.path.join(directory, "driver_N*.log"))):
            worker_log = driver_log.replace("/driver_N", "/worker_N")
            if not os.path.exists(worker_log):
                continue
            n = int(re.search(r"_N(\d+)", os.path.basename(driver_log)).group(1))
            builds = builds_of(driver_log)
            ranks, fmt = ranks_of(worker_log, kind)
            if not builds or not ranks or fmt is None:
                continue
            per = max(len(ranks) // len(builds), 1)   # matrices reported per build
            for i, (hyper, _) in enumerate(builds):
                window = ranks[i * per:(i + 1) * per]
                if not window:
                    continue
                metrics = hyper[3:]                   # one entry per length scale
                # an isotropic kernel has a single length scale; an anisotropic one is summarised
                # by the geometric mean, which is what sets the volume of a correlated neighbourhood
                lengthscale = math.exp(sum(metrics) / (2.0 * len(metrics)))
                density = n * lengthscale ** len(metrics)
                if density > 0:
                    points[fmt].append((density, max(window)))
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
