#! /usr/bin/env python
"""
Sanity checks of the replicas of kaust2022_2b_replica.py and length-scale search boxes for an anisotropic GP over
(x, y, t_norm), t_norm = slot / 99:

  * empirical semivariograms of the training values: spatial (pairs of sites within the same slot, summed over
    all the slots, by distance bins) and temporal (same site, by lag), against the true C(0,0) - C(h,0) (averaged
    over the pairs of each bin) and C(0,0) - C(0,u) (plus the nugget at nonzero lags);
  * effective ranges (where the correlation 1 - gamma / sill drops to 0.05), empirical with the true sill and
    with the sample variance as the sill, and true, in normalized units (space in [0, 1], time in t_norm);
  * maximum-likelihood fits of an exact GP, a constant times an axis-aligned RBF exp(-r^2/2) or Matern-3/2
    (1 + sqrt(3) r) exp(-sqrt(3) r), r^2 = sum_d (dx_d / l_d)^2 (george's length scales, metric l^2), plus a
    noise variance, to the training values minus their mean in a space-time window at the full data density
    (--window-side x --window-slots), as a guide for the initial length scales.

Usage: python kaust2022_2b_variogram.py [npz ...]
  (default: the RS replicas kaust2022_2b{1,2,3}_replica.npz and kaust2022_2b2_replica_nug0.01.npz in
   $PSCRATCH/kaust2022 that exist; --no-fit skips the GP fits)
Writes <npz>_variogram.npz next to each input (bin centres, counts, empirical and true semivariograms).
Run it on a compute node (the GP fits factor ~5000 x 5000 matrices ~150 times each: ~9 minutes per file;
--no-fit: ~1 minute), e.g.
  sbatch -A m2957 -q debug -C cpu -N 1 -t 00:30:00 --wrap "module load python; python kaust2022_2b_variogram.py"
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from kaust2022_2b_replica import gneiting, NSLOTS

EDGES = np.concatenate([np.arange(0, 0.1, 0.0025), np.arange(0.1, 1.5 + 1e-9, 0.01)])


def to_matrix(d):
    """Training values as an (nsites, nslots) array with a mask (sites renumbered 0..n-1)."""
    sites, inv = np.unique(d['site_train'], return_inverse=True)
    Z = np.zeros((len(sites), NSLOTS))
    M = np.zeros((len(sites), NSLOTS))
    Z[inv, d['slot_train']] = d['ytrain']
    M[inv, d['slot_train']] = 1
    xy = np.zeros((len(sites), 2))
    xy[inv] = d['xtrain'][:, :2]
    return Z, M, xy


def spatial_variogram(Z, M, xy, p, tau2, block=500):
    """Binned sums over same-slot pairs: counts, sum (z_i - z_j)^2, sum of the true semivariogram."""
    A = (M * Z ** 2) @ M.T
    B = (M * Z) @ (M * Z).T
    N = M @ M.T
    nb = len(EDGES) - 1
    cnt, ssq, hsum, gtrue = (np.zeros(nb) for _ in range(4))
    C0 = gneiting(0.0, 0.0, **p)
    n = len(xy)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d = cdist(xy[i0:i1], xy)
        rows, cols = np.nonzero(np.arange(n)[None, :] > np.arange(i0, i1)[:, None])
        dd = d[rows, cols]
        ii = rows + i0
        b = np.searchsorted(EDGES, dd, side='right') - 1
        keep = (b >= 0) & (b < nb)
        b, dd, ii, jj = b[keep], dd[keep], ii[keep], cols[keep]
        w = N[ii, jj]
        cnt += np.bincount(b, w, nb)
        ssq += np.bincount(b, A[ii, jj] + A[jj, ii] - 2 * B[ii, jj], nb)
        hsum += np.bincount(b, w * dd, nb)
        gtrue += np.bincount(b, w * (C0 + tau2 - gneiting(dd, 0.0, **p)), nb)
    ok = cnt > 0
    h = np.where(ok, hsum / np.maximum(cnt, 1), 0.5 * (EDGES[:-1] + EDGES[1:]))
    return h[ok], cnt[ok], ssq[ok] / (2 * cnt[ok]), gtrue[ok] / cnt[ok]


def temporal_variogram(Z, M, p, tau2):
    lags = np.arange(1, NSLOTS)
    emp, cnt = np.zeros(len(lags)), np.zeros(len(lags))
    for k, u in enumerate(lags):
        w = M[:, u:] * M[:, :-u]
        cnt[k] = w.sum()
        emp[k] = (w * (Z[:, u:] - Z[:, :-u]) ** 2).sum() / (2 * cnt[k])
    true = gneiting(0.0, 0.0, **p) + tau2 - gneiting(0.0, lags.astype(float), **p)
    return lags, cnt, emp, true


def eff_range(x, gamma, sill):
    """First x where 1 - gamma / sill <= 0.05 (linear interpolation), or nan if not reached."""
    rho = 1 - gamma / sill
    below = np.flatnonzero(rho <= 0.05)
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return x[0]
    return x[k - 1] + (rho[k - 1] - 0.05) / (rho[k - 1] - rho[k]) * (x[k] - x[k - 1])


def gp_nll(theta, X, y, kern):
    """Negative log marginal likelihood and its gradient in theta = log(sf2, l_1..l_D, sn2)."""
    sf2, ls, sn2 = np.exp(theta[0]), np.exp(theta[1:-1]), np.exp(theta[-1])
    D2 = [((X[:, None, d] - X[None, :, d]) / ls[d]) ** 2 for d in range(X.shape[1])]
    r2 = sum(D2)
    if kern == 'RBF':
        k = np.exp(-0.5 * r2)
        dk = [sf2 * k * q for q in D2]
    else:
        r = np.sqrt(r2)
        e = np.exp(-np.sqrt(3) * r)
        k = (1 + np.sqrt(3) * r) * e
        dk = [3 * sf2 * e * q for q in D2]
    K = sf2 * k
    K[np.diag_indices_from(K)] += sn2
    try:
        cf = la.cho_factor(K, lower=True, check_finite=False)
    except la.LinAlgError:
        return 1e25, np.zeros_like(theta)
    alpha = la.cho_solve(cf, y, check_finite=False)
    nll = 0.5 * y @ alpha + np.log(np.diag(cf[0])).sum() + 0.5 * len(y) * np.log(2 * np.pi)
    W = np.outer(alpha, alpha) - la.cho_solve(cf, np.eye(len(y)), check_finite=False)
    grad = [-0.5 * np.sum(W * (sf2 * k))] + [-0.5 * np.sum(W * g) for g in dk] + [-0.5 * sn2 * np.trace(W)]
    return nll, np.array(grad)


def window_fit(d, kern, side, nslots, center=(0.5, 0.5, 50)):
    X, y, slot = d['xtrain'], d['ytrain'], d['slot_train']
    s0 = int(center[2] - nslots // 2)
    m = ((np.abs(X[:, 0] - center[0]) < side / 2) & (np.abs(X[:, 1] - center[1]) < side / 2)
         & (slot >= s0) & (slot < s0 + nslots))
    Xw, yw = X[m], y[m] - y[m].mean()
    best = None
    for l0 in ((0.01, 0.01, 0.02), (0.05, 0.05, 0.1), (0.02, 0.02, 0.3)):
        theta0 = np.log([yw.var(), *l0, 0.01 * yw.var()])
        bounds = [(np.log(1e-3), np.log(1e2))] + [(np.log(1e-4), np.log(1e2))] * 3 + [(np.log(1e-8), np.log(1e1))]
        res = minimize(gp_nll, theta0, args=(Xw, yw, kern), jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=200))
        if best is None or res.fun < best.fun:
            best = res
    t = np.exp(best.x)
    return dict(kern=kern, n=int(m.sum()), window=dict(side=side, slots=[s0, s0 + nslots]), amplitude=t[0],
                l_x=t[1], l_y=t[2], l_t=t[3], noise=t[4], nll=float(best.fun), converged=bool(best.success), nit=int(best.nit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--no-fit', action='store_true')
    ap.add_argument('--window-side', type=float, default=0.2)
    ap.add_argument('--window-slots', type=int, default=14)
    args = ap.parse_args()
    base = os.path.join(os.environ.get('PSCRATCH', '.'), 'kaust2022')
    files = args.files or [f for f in (os.path.join(base, 'kaust2022_2b%s_replica%s.npz' % s) for s in
                                       (('1', ''), ('2', ''), ('2', '_nug0.01'), ('3', ''))) if os.path.exists(f)]
    for f in files:
        t0 = time.time()
        d = np.load(f)
        meta = json.loads(str(d['meta']))
        p = dict(a_s=meta['a_s'], a_t=meta['a_t'], alpha=meta['alpha'], sigma2=meta['sigma2'], beta=meta['beta'], nu=meta['nu'])
        tau2 = meta['nugget']
        Z, M, xy = to_matrix(d)
        nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
        sill_true, sill_emp = p['sigma2'] + tau2, d['ytrain'].var()
        h, hc, hemp, htrue = spatial_variogram(Z, M, xy, p, tau2)
        lags, lc, temp, ttrue = temporal_variogram(Z, M, p, tau2)
        tn = lags / (NSLOTS - 1.0)
        res = dict(file=os.path.basename(f), setting=meta['setting'], scheme=meta['scheme'], nugget=tau2,
                   sites=len(xy), nn_spacing_median=float(np.median(nn)), nn_spacing_min=float(nn.min()),
                   train_variance=float(sill_emp),
                   range_space=dict(true=meta['effective_range_space'], true_binned=eff_range(h, htrue, sill_true),
                                    emp_true_sill=eff_range(h, hemp, sill_true), emp_sample_sill=eff_range(h, hemp, sill_emp)),
                   range_time_tnorm=dict(true=meta['effective_range_time_tnorm'], emp_true_sill=eff_range(tn, temp, sill_true),
                                         emp_sample_sill=eff_range(tn, temp, sill_emp)))
        print("\n== %s (%s %s, nugget %g): %d sites, NN spacing median %.4f min %.4f; training variance %.4f (sill %.3f)"
              % (res['file'], res['setting'], res['scheme'], tau2, len(xy), res['nn_spacing_median'], res['nn_spacing_min'],
                 sill_emp, sill_true))
        print("spatial semivariogram (same slot):   h     pairs      empirical  true")
        for target in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.3):
            k = np.argmin(np.abs(h - target))
            print("%40.4f %10.3g %10.4f %10.4f" % (h[k], hc[k], hemp[k], htrue[k]))
        print("temporal semivariogram (same site):  lag (t_norm)  empirical  true")
        for u in (1, 2, 3, 5, 10, 20, 40, 60, 80, 99):
            print("%34d (%.4f) %10.4f %10.4f" % (u, tn[u - 1], temp[u - 1], ttrue[u - 1]))
        print("effective ranges: space %s; time (t_norm) %s"
              % (json.dumps({k: round(v, 4) for k, v in res['range_space'].items()}),
                 json.dumps({k: round(v, 4) for k, v in res['range_time_tnorm'].items()})))
        print("relative RMS deviation empirical/true: space (h <= true range) %.3f, time (lag <= 20) %.3f"
              % (np.sqrt(np.mean((hemp[h <= res['range_space']['true']] / htrue[h <= res['range_space']['true']] - 1) ** 2)),
                 np.sqrt(np.mean((temp[:20] / ttrue[:20] - 1) ** 2))), flush=True)
        if not args.no_fit:
            res['fits'] = []
            for kern in ('RBF', 'Matern32'):
                t1 = time.time()
                fit = window_fit(d, kern, args.window_side, args.window_slots)
                res['fits'].append(fit)
                print("%-8s window fit (%d points, side %.2f, slots %s): l_x %.4g l_y %.4g l_t %.4g (t_norm; %.2f slots),"
                      " amplitude %.4g, noise %.3g, converged %s, %d it, %.0f s"
                      % (kern, fit['n'], args.window_side, fit['window']['slots'], fit['l_x'], fit['l_y'], fit['l_t'],
                         fit['l_t'] * (NSLOTS - 1), fit['amplitude'], fit['noise'], fit['converged'], fit['nit'],
                         time.time() - t1), flush=True)
        out = f.replace('.npz', '_variogram.npz')
        np.savez(out, h=h, h_pairs=hc, h_emp=hemp, h_true=htrue, lags=lags, lag_pairs=lc, t_emp=temp, t_true=ttrue,
                 summary=json.dumps(res))
        print("wrote %s (%.0f s)" % (out, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
