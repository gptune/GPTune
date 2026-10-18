#! /usr/bin/env python
"""
Rough synthetic Gaussian-process data on the (clustered) designs of datasets 2 and 4 of sub-competition 2b of the
2023 KAUST Competition on Spatial Statistics for Large Datasets (npz files of kaust2023_2b_npz.py).  The original
fields are smooth (Matern nu 2.3 / 2.1, effective range ~0.3), so they do not test whether a model resolves the
fine scales sampled by the clusters; here the same training and testing locations (train, test set 1 and test
set 2, in the same order) get a stationary isotropic Matern field

    C(r) = sigma2 M_nu(r / ell),   M_nu(x) = x^nu K_nu(x) / (2^(nu-1) Gamma(nu)),   sigma2 = 1,

with effective range 0.02 (C(0.02) = 0.05 sigma2: ell = 0.02 / 3.00 for nu = 0.5, 0.02 / 4.00 for nu = 1), plus
N(0, tau2) noise, tau2 = 0.01, on all the observed values (training and testing); the noise-free testing values
are kept in mtest and mtest2.

Simulation: 2D circulant embedding on a regular grid of spacing h = 2 / L (default L = 65536, h = 1/32768) of
the periodic domain [0, 2)^2; the eigenvalues are the 2D DCT-I of one quadrant of the base covariance (truncated
to zero beyond 50 ell, where C < 1e-20), the negative ones (round-off) are clipped and reported.  The real and
imaginary parts of one complex FFT are two independent fields, used for dataset 2 and dataset 4; the complex
white noise is the same for all nu (common random numbers) and the observation noise is the same for all nu.
Each location takes the value of its nearest grid node (indices taken modulo L: dataset 4 extends slightly
outside [0, 1]^2, to about [-0.04, 1.05]^2, and its points are at least 2 - 1.1 = 0.9 apart through the periodic
boundary, where C = 0).  Approximation: the field values are exact (up to the embedding) at the snapped
locations round(x / h) h, not at x; the script reports the maximum snapping distance and the mean variance of
Z(x) - Z(node), 2 (1 - rho(|x - node|)), which is the extra noise a model of the exact locations sees.

Usage (on a CPU compute node; L = 65536: measured peak memory ~80 GB, 80 s for both nu and both datasets):
  python kaust2023_rough_matern.py [--L 65536] [--nus 0.5,1.0] [--datasets 2,4]
  e.g. sbatch -A m2957 -q debug -C cpu -N 1 -t 00:30:00 --wrap "module load python; python kaust2023_rough_matern.py"
Reads $PSCRATCH/kaust2023/kaust2023_2b_dataset0<d>.npz and writes $PSCRATCH/kaust2023/rough_ds<d>_nu<nu>_r0.02.npz
with xtrain, ytrain, xtest, ytest, mtest, xtest2, ytest2, mtest2 and meta (json).
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft
import scipy.special as sp
from scipy.optimize import brentq
from scipy.spatial import cKDTree

NTHREADS = os.cpu_count()
SEED = 2023
PERIOD = 2.0


def matern(x, nu):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    pos = x > 0
    xp = x[pos]
    if nu == 0.5:
        out[pos] = np.exp(-xp)
    elif nu == 1.0:
        out[pos] = xp * sp.k1(xp)
    else:
        out[pos] = xp ** nu * sp.kv(nu, xp) / (2 ** (nu - 1) * sp.gamma(nu))
    return out


def ell_from_range(eff_range, nu):
    return eff_range / brentq(lambda x: matern(np.array([x]), nu)[0] - 0.05, 1e-3, 100)


def quadrant_eigenvalues(nu, ell, sigma2, L, h):
    """Quadrant (L/2+1)^2 of the circulant-embedding eigenvalues and diagnostics."""
    t0 = time.time()
    M = L // 2 + 1
    K = min(M, int(np.ceil(50 * ell / h)) + 1)
    c = np.zeros((M, M))
    hx = h * np.arange(K)

    def rows(i0):
        i1 = min(i0 + 256, K)
        r = np.sqrt(hx[i0:i1, None] ** 2 + hx[None, :] ** 2)
        c[i0:i1, :K] = sigma2 * matern(r / ell, nu)
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(rows, range(0, K, 256)))
    lam = scipy.fft.dctn(c, type=1, workers=NTHREADS, overwrite_x=True)
    del c
    m = np.full(M, 2.0)
    m[0] = m[-1] = 1
    neg = lam < 0
    w_neg = float((m[:, None] * np.where(neg, -lam, 0) * m[None, :]).sum())
    w_pos = float((m[:, None] * np.where(neg, 0, lam) * m[None, :]).sum())
    diag = dict(L=L, h=h, period=L * h, truncation_radius=K * h, lambda_max=float(lam.max()),
                lambda_min=float(lam.min()), min_over_max=float(lam.min() / lam.max()),
                n_negative=int((m[:, None] * neg * m[None, :]).sum()), neg_mass_fraction=w_neg / (w_neg + w_pos),
                variance_from_trace=(w_pos - w_neg) / L ** 2, variance_after_clipping=w_pos / L ** 2,
                seconds=time.time() - t0)
    return lam, diag


def fill_normal(a, seed, nchunks=256):
    gens = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(nchunks)]
    bounds = np.linspace(0, a.shape[0], nchunks + 1).astype(int)

    def chunk(i):
        if bounds[i + 1] > bounds[i]:
            gens[i].standard_normal(out=a[bounds[i]:bounds[i + 1]])
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(chunk, range(nchunks)))


def simulate(lam, L, seed):
    """Complex field on the periodic L x L grid (real and imaginary parts: two independent fields)."""
    np.maximum(lam, 0, out=lam)
    lam /= float(L) ** 2
    amp = np.sqrt(lam, out=lam)
    mirror = np.minimum(np.arange(L), L - np.arange(L))
    w = np.empty((L, L), dtype=np.complex128)
    fill_normal(w.view(np.float64), seed)

    def scale(i0):
        for i in range(i0, min(i0 + 64, L)):
            w[i] *= amp[mirror[i]][mirror]
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(scale, range(0, L, 64)))
    return scipy.fft.fft2(w, workers=NTHREADS, overwrite_x=True)


def nn_spacing(x):
    """Median nearest-neighbour distances overall and by local density (50 x 50 bins over [-0.05, 1.05]^2)."""
    nn = cKDTree(x).query(x, k=2, workers=NTHREADS)[0][:, 1]
    nb, lo, width = 50, -0.05, 1.1
    idx = np.clip(((x - lo) / width * nb).astype(int), 0, nb - 1)
    H = np.zeros((nb, nb))
    np.add.at(H, (idx[:, 0], idx[:, 1]), 1)
    rel = H[idx[:, 0], idx[:, 1]] / (width / nb) ** 2 / len(x)  # local density relative to uniform on [0,1]^2
    out = dict(median_nn_all=float(np.median(nn)), nn_percentiles_10_50_90=np.percentile(nn, [10, 50, 90]).tolist())
    for name, m in (('dense_rel_density_ge_5', rel >= 5), ('sparse_rel_density_lt_1', rel < 1),
                    ('in_0_0.5_square', (x >= 0).all(1) & (x < 0.5).all(1)),
                    ('outside_0_0.5_square', ~((x >= 0).all(1) & (x < 0.5).all(1)))):
        out[name] = dict(fraction=float(m.mean()), median_nn=float(np.median(nn[m])) if m.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--L', type=int, default=65536, help='periodic grid size of [0, 2)^2 (spacing 2/L)')
    ap.add_argument('--nus', default='0.5,1.0')
    ap.add_argument('--datasets', default='2,4')
    ap.add_argument('--range', type=float, default=0.02, help='effective range (correlation 0.05)')
    ap.add_argument('--sigma2', type=float, default=1.0)
    ap.add_argument('--tau2', type=float, default=0.01)
    ap.add_argument('--dir', default=os.path.join(os.environ.get('PSCRATCH', '.'), 'kaust2023'), help='input directory')
    ap.add_argument('--outdir', default='', help='output directory (default --dir)')
    args = ap.parse_args()
    args.outdir = args.outdir or args.dir
    L, h = args.L, PERIOD / args.L
    datasets = [int(d) for d in args.datasets.split(',')]
    assert len(datasets) <= 2, 'one complex FFT gives two independent fields'
    src = {d: np.load(os.path.join(args.dir, 'kaust2023_2b_dataset%02d.npz' % d)) for d in datasets}
    locs, nodes, info = {}, {}, {}
    for d in datasets:
        locs[d] = {k: src[d][k] for k in ('xtrain', 'xtest', 'xtest2')}
        nodes[d] = {k: np.mod(np.rint(v / h).astype(np.int64), L) for k, v in locs[d].items()}
        snap = {k: np.hypot(*(locs[d][k] - np.rint(locs[d][k] / h) * h).T) for k in locs[d]}
        tr = nodes[d]['xtrain'][:, 0] * L + nodes[d]['xtrain'][:, 1]
        _, cnt = np.unique(tr, return_counts=True)
        info[d] = dict(snap=snap, spacing=nn_spacing(locs[d]['xtrain']),
                       train_points_sharing_a_node=float(cnt[cnt > 1].sum() / len(tr)),
                       extent=[locs[d]['xtrain'].min(0).tolist(), locs[d]['xtrain'].max(0).tolist()])
        s = info[d]['spacing']
        print("dataset %d: training extent %s; median NN distance %.2e overall, %.2e in dense regions (local density"
              " >= 5x uniform, %.0f%% of points), %.2e in sparse regions (< 1x, %.0f%%); [0,0.5]^2: %.2e (%.0f%%),"
              " outside: %.2e; %.1f%% of training points share their grid node"
              % (d, info[d]['extent'], s['median_nn_all'], s['dense_rel_density_ge_5']['median_nn'],
                 100 * s['dense_rel_density_ge_5']['fraction'], s['sparse_rel_density_lt_1']['median_nn'],
                 100 * s['sparse_rel_density_lt_1']['fraction'], s['in_0_0.5_square']['median_nn'],
                 100 * s['in_0_0.5_square']['fraction'], s['outside_0_0.5_square']['median_nn'],
                 100 * info[d]['train_points_sharing_a_node']), flush=True)
    for nu in (float(v) for v in args.nus.split(',')):
        ell = ell_from_range(args.range, nu)
        lam, diag = quadrant_eigenvalues(nu, ell, args.sigma2, L, h)
        print("nu %.1f ell %.6f: embedding %d^2 (h %.3e, period %.1f): lambda_min/lambda_max %.3e, %d negative,"
              " negative mass fraction %.3e, variance %.6f -> %.6f after clipping (%.1f s)"
              % (nu, ell, L, h, L * h, diag['min_over_max'], diag['n_negative'], diag['neg_mass_fraction'],
                 diag['variance_from_trace'], diag['variance_after_clipping'], diag['seconds']), flush=True)
        t0 = time.time()
        z = simulate(lam, L, SEED)
        del lam
        # sanity check on the [0, 1]^2 part of the grid: variance and correlation at lags ell, 3 ell along x
        win = z[:L // 2:4, :L // 2]
        check = {}
        for part, f in (('real', win.real), ('imag', win.imag)):
            v = f.var()
            check[part] = dict(variance=float(v))
            for mult in (1, 3):
                k = int(round(mult * ell / h))
                check[part]['corr_at_%dell' % mult] = float(np.mean(f[:, k:] * f[:, :-k]) / v)
        check['true_corr_at_1ell'], check['true_corr_at_3ell'] = (float(matern(np.array([k * h / ell]), nu)[0])
                                                                  for k in (round(ell / h), round(3 * ell / h)))
        print("  simulated in %.1f s; grid check %s" % (time.time() - t0, check), flush=True)
        for j, d in enumerate(datasets):
            f = z.real if j == 0 else z.imag
            rng = np.random.default_rng(SEED + d)
            vals = {k: np.ascontiguousarray(f[nodes[d][k][:, 0], nodes[d][k][:, 1]]) for k in locs[d]}
            noisy = {k: v + np.sqrt(args.tau2) * rng.standard_normal(len(v)) for k, v in vals.items()}
            snap = info[d]['snap']
            dmax = max(s.max() for s in snap.values())
            mismatch = {k: float(np.mean(2 * args.sigma2 * (1 - matern(s / ell, nu)))) for k, s in snap.items()}
            meta = dict(description='rough Matern field on the locations of KAUST 2023 sub-competition 2b dataset %d'
                        ' (kaust2023_rough_matern.py)' % d, source=os.path.join(args.dir, 'kaust2023_2b_dataset%02d.npz' % d),
                        model='sigma2 M_nu(r/ell) + N(0, tau2)', nu=nu, ell=ell, effective_range=args.range,
                        sigma2=args.sigma2, tau2=args.tau2, grid_spacing=h, grid_L=L, period=PERIOD,
                        snapped_location='round(x / grid_spacing) * grid_spacing', embedding=diag,
                        field='real part' if j == 0 else 'imaginary part',
                        seeds=dict(white_noise=SEED, observation_noise=SEED + d),
                        max_snap_distance=float(dmax), max_snap_over_range=float(dmax / args.range),
                        mean_snap_mismatch_variance=mismatch,
                        train_points_sharing_a_node=info[d]['train_points_sharing_a_node'],
                        nn_spacing=info[d]['spacing'], grid_check=check,
                        sample_variance_noise_free_train=float(vals['xtrain'].var()))
            out = os.path.join(args.outdir, 'rough_ds%d_nu%.1f_r%g.npz' % (d, nu, args.range))
            np.savez(out, xtrain=locs[d]['xtrain'], ytrain=noisy['xtrain'], xtest=locs[d]['xtest'],
                     ytest=noisy['xtest'], mtest=vals['xtest'], xtest2=locs[d]['xtest2'], ytest2=noisy['xtest2'],
                     mtest2=vals['xtest2'], meta=json.dumps(meta))
            print("  %s: train %d test %d test2 %d; noise-free train variance %.4f; max snap distance %.2e (%.2e of"
                  " the range); mean snap mismatch variance %s"
                  % (out, len(vals['xtrain']), len(vals['xtest']), len(vals['xtest2']), vals['xtrain'].var(), dmax,
                     dmax / args.range, {k: '%.2e' % v for k, v in mismatch.items()}), flush=True)
        del z, win, f


if __name__ == '__main__':
    main()
