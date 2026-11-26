"""The numbers of the paper's Table III (tab:wend_opt): Wendland against INLA, the lowest final -log likelihood and
the lowest final test RMSE (1,000 test points) among the training algorithms run at each N, read from the
training_metrics_obj<o>_N<N>_<optimizer>.csv files (last row: the returned hyperparameters, i.e. the last L-BFGS
iterate or the best MCMC/MALA sample; RMSE of the last replayed history entry).
  2D Schwefel (objective 4), N = 100k, 400k, 1.6M:
    Wendland  the runs of the former Table III: scaling_obj4_reuse (gradient, FD, capped MCMC), scaling_obj4_stagnation
              (MCMC without cap, 100k), scaling_obj4_mala_stagnation (MALA, 100k and 400k)
    INLA      inla_scaling_obj4/inla (nu = 1, sqrt(N) lattice), all four optimizers
  3D isotropic quadratic (bpack objective 3), N = 100k, 400k, 800k, L-BFGS only:
    Wendland  $WENDLAND_3D/wendland_N<N> (default sparse_3d_obj3, the zero-mean runs; sparse_3d_obj3_mean: sample mean)
    INLA      nu = 0.5: sparse_3d_obj3/inla_nu0.5_N<N> (gradient and FD), and the earlier gradient runs
              inla_3d/obj3_N<N>_nu0.5 where sparse_3d_obj3 has no run of that optimizer"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMIZERS = ("gradient", "finite_difference", "mcmc", "mala")
# the 3D Wendland runs: sparse_3d_obj3 (zero prior mean, jobs 61771465/66/91) or sparse_3d_obj3_mean (the prior mean
# set to the sample mean of the training outputs, GP_MEAN=sample, as the INLA model; jobs 61923410/12)
WENDLAND_3D = os.environ.get("WENDLAND_3D", "sparse_3d_obj3")


def final_metrics(directory, objtype, N, optimizers=OPTIMIZERS):
    """{optimizer: (final -log likelihood, final test RMSE)} of the runs whose training_metrics file is under
    the directory"""
    out = {}
    for optimizer in optimizers:
        paths = sorted(glob.glob(os.path.join(directory, "**", "training_metrics_obj%d_N%d_%s.csv" % (objtype, N, optimizer)),
                                 recursive=True))
        if not paths:
            continue
        a = np.atleast_1d(np.genfromtxt(paths[0], delimiter=",", names=True))
        rmse = a["rmse"][np.isfinite(a["rmse"])]
        out[optimizer] = (float(a["nll"][-1]), float(rmse[-1]) if rmse.size else np.nan, os.path.relpath(paths[0], HERE))
    return out


def sources(case, model, N):
    d = lambda *p: os.path.join(HERE, *p)
    if case == "2D Schwefel":
        if model == "Wendland":
            runs = [(d("scaling_obj4_reuse"), ("gradient", "finite_difference", "mcmc"))]
            if N == 100000:
                runs.append((d("scaling_obj4_stagnation"), ("mcmc",)))
            if N in (100000, 400000):
                runs.append((d("scaling_obj4_mala_stagnation"), ("mala",)))
            return 4, runs
        return 4, [(d("inla_scaling_obj4", "inla"), OPTIMIZERS)]
    if model == "Wendland":
        return 3, [(d(WENDLAND_3D, "wendland_N%d" % N), OPTIMIZERS)]
    new = d("sparse_3d_obj3", "inla_nu0.5_N%d" % N)
    have = set(final_metrics(new, 3, N))
    return 3, [(new, OPTIMIZERS), (d("inla_3d", "obj3_N%d_nu0.5" % N), tuple(o for o in OPTIMIZERS if o not in have))]


CASES = [("2D Schwefel", N) for N in (100000, 400000, 1600000)] + [("3D isotropic", N) for N in (100000, 400000, 800000)]
if __name__ == "__main__":
    for case, N in CASES:
        cells = []
        for model in ("Wendland", "INLA"):
            objtype, runs = sources(case, model, N)
            found = {}
            for directory, optimizers in runs:
                for o, v in final_metrics(directory, objtype, N, optimizers).items():
                    found.setdefault(o if o not in found else o + " (" + os.path.basename(directory) + ")", v)
            if not found:
                cells.append("%s: no runs yet" % model)
                continue
            nll = min(v[0] for v in found.values()); rmse = np.nanmin([v[1] for v in found.values()])
            arg_nll = [o for o, v in found.items() if v[0] == nll][0]
            arg_rmse = [o for o, v in found.items() if v[1] == rmse][0]
            cells.append("%s: min -L %.2f (%s)  min RMSE %.3e (%s)" % (model, nll, arg_nll, rmse, arg_rmse))
            for o, v in found.items():
                print("    %-9s %-13s N=%-8d %-30s -L %16.2f  RMSE %.3e  %s" % (model, case, N, o, v[0], v[1], v[2]))
        print("%-13s N=%-8d %s" % (case, N, " | ".join(cells)))
