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

# N = 200k block of Table II: per objective and format, the largest factorization rank over the builds of all 200k
# runs (all four optimizers) and the lowest -log likelihood any of them recorded (training_iterations: L-BFGS
# iterates, MCMC/MALA best samples). A recorded value below N/2 log(sigma^2) (hp_0 = log sigma^2 of that row) is
# impossible, since K = K_f + sigma^2 I with K_f positive semidefinite, and is dropped (broken 1D oscillatory runs).
N200 = 200000
RUNS200 = [("1D oscillatory", ["scal_obj1_h2cpu"], ["scal_obj1_hodlrcpu", "scal_obj1ls_hodlrcpu"]),  # CPU, Figs. 2a/4a
           ("1D Schwefel", ["scal_1d_h2gpu"], ["scal_1d_hodlrgpu"]),
           ("2D isotropic", ["scal_h2gpu_2d_iso"], ["scal_hodlrgpu_2d_iso"]),
           ("2D anisotropic", ["scal_h2gpu_2d_aniso_scaled"], ["scal_hodlrgpu_2d_aniso_scaled"]),
           ("3D isotropic", ["scal_h2gpu_3d_iso"], []),
           ("3D anisotropic", ["scal_h2gpu_3d_aniso"], [])]  # raw coordinates (dagger in Fig. 4d); no FD worker log
SUFFIX = {"gradient": "grad", "finite_difference": "fd", "mcmc": "mcmc", "mala": "mala"}
def run_logs(run):
    """{optimizer: worker log} of the 200k runs in a run directory; the unsuffixed log belongs to the optimizer
    whose training_iterations file its driver log names"""
    out = {o: os.path.join(C, run, "worker_N%d_%s.log" % (N200, s)) for o, s in SUFFIX.items()}
    driver = os.path.join(C, run, "driver_N%d.log" % N200)
    if os.path.exists(driver):
        m = re.search(r"training_iterations_obj\d+_N\d+_([a-z_]+)\.csv", open(driver, errors="replace").read())
        if m and not os.path.exists(out[m.group(1)]):
            out[m.group(1)] = os.path.join(C, run, "worker_N%d.log" % N200)
    return {o: f for o, f in out.items() if os.path.exists(f)}
def best_nll(run):
    best, dropped = None, []
    for f in sorted(glob.glob(os.path.join(C, run, "training_iterations_obj*_N%d_*.csv" % N200))):
        a = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
        if not a.size or a.dtype.names is None:
            continue
        ok = a["nll"] >= 0.5 * N200 * a["hp_0"]
        dropped += ["%s %.4g" % (os.path.basename(f).split("_N%d_" % N200)[1][:-4], v) for v in a["nll"][~ok]]
        if ok.any():
            best = min(best, a["nll"][ok].min()) if best is not None else a["nll"][ok].min()
    return best, dropped
print("\nN = 200k: max rank over all builds, min -log likelihood over all recorded values")
for name, h2, hodlr in RUNS200:
    cells = []
    for runs in (h2, hodlr):
        if not runs:
            cells.append("%5s %15s" % ("-", "-")); continue
        ranks = [x[1] for run in runs for f in run_logs(run).values() for x in k_builds(f)[1]]
        nll = [best_nll(run) for run in runs]
        best = min(b for b, _ in nll if b is not None)
        dropped = [d for _, ds in nll for d in ds]
        cells.append("%5d %15.2f" % (max(ranks), best) + ("  (dropped %s)" % ", ".join(dropped) if dropped else ""))
    print("%-15s H2 %s | HODLR %s" % (name, cells[0], cells[1]))
