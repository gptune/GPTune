"""The numbers of the paper's Table II (tab:hacc) and of the text on Fig. 2, a build = one H2/HODLR matrix of K built and factored at one
hyperparameter value (plot_rank_vs_density.k_builds). Table IX runs: the optimizer logs worker_N<n>_<opt>.log
of the rows of tab:h2gpu."""
import os, re, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_rank_vs_density import k_builds
C = os.path.join(HERE, 'h2_unstructured_cmp')
T9 = {7: {"2D iso": [("scal_h2gpu_2d_iso", [100000, 200000, 400000])],
          "2D aniso": [("scal_h2gpu_2d_aniso_scaled", [100000, 200000]), ("scal_h2gpu_2d_aniso", [400000])],
          "3D iso": [("scal_h2gpu_3d_iso", [100000, 200000])],
          "3D aniso": [("scal_h2gpu_3d_aniso_scaled", [100000]), ("scal_h2gpu_3d_aniso", [200000])]},
      1: {"2D iso": [("scal_hodlrgpu_2d_iso", [100000, 200000])],
          "2D aniso": [("scal_hodlrgpu_2d_aniso_scaled", [100000, 200000])],
          "3D aniso": [("scal_hodlrgpu_3d_aniso_scaled", [25000])]}}
def logs(run, Ns):
    return [f for n in Ns for f in sorted(glob.glob(os.path.join(C, run, "worker_N%d_*.log" % n)))]
for fmt, cases in T9.items():
    tot = []
    for case, rows in cases.items():
        b = [x for run, Ns in rows for f in logs(run, Ns) for x in k_builds(f)[1]]
        r = np.array([x[1] for x in b]); e = np.array([x[2] for x in b]); tot += b
        print("%s %-8s builds %5d  error median %.2e max %.2e >1e-3 %3d (%.1f%%)  rank median %5d max %5d" % ("H2   " if fmt == 7 else "HODLR", case, len(b), np.median(e), e.max(), (e > 1e-3).sum(), 100 * (e > 1e-3).mean(), np.median(r), r.max()))
    e = np.array([x[2] for x in tot]); print("%s TOTAL    builds %5d  error median %.2e max %.2e >1e-3 %d" % ("H2   " if fmt == 7 else "HODLR", len(e), np.median(e), e.max(), (e > 1e-3).sum()))
# 1D rows of Table II: oscillatory = Fig. 2a CPU runs (without the duplicate 1.6M log), Schwefel = 1D GPU runs
for name, runs in [("1D osc CPU (Fig 2a)", [("scal_obj1_h2cpu", 7), ("scal_obj1_hodlrcpu", 1)]), ("1D Schwefel GPU", [("scal_1d_h2gpu", 7), ("scal_1d_hodlrgpu", 1)])]:
    for run, fmt in runs:
        fs = [f for f in sorted(glob.glob(os.path.join(C, run, "worker_N*.log"))) if not f.endswith("scal_obj1_h2cpu/worker_N1600000.log")]
        r = np.array([x[1] for f in fs for x in k_builds(f)[1]])
        print("%-20s %-18s builds %5d rank median %d max %d" % (name, run, len(r), np.median(r), r.max()))
