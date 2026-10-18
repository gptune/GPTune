#! /usr/bin/env python
"""
A replica of sub-competition 2b (space-time) of the 2022 KAUST Competition on Spatial Statistics for Large
Datasets (Abdulah et al., "The Second Competition on Spatial Statistics for Large Datasets", arXiv:2211.03119,
Sec. 3.2 and Table 1): a zero-mean Gaussian process at 10,000 sites of [0, 1]^2 x 100 time slots with the
non-separable stationary space-time covariance of Gneiting (2002)

    C(h, u) = sigma2 / psi(u) * M_nu( ||h|| / (a_s psi(u)^(beta/2)) ),   psi(u) = a_t |u|^(2 alpha) + 1,
    M_nu(r) = r^nu K_nu(r) / (2^(nu-1) Gamma(nu)),   M_nu(0) = 1,

sigma2 = 0.9, beta = 0.9, nu = 1, u in time-slot units, no nugget, and three settings (space / time effective
ranges, the distances where the correlation is 0.05):

    weak-strong        a_s 0.02  a_t 1     alpha 0.6    0.08 / 11.63 slots   datasets 2b-1 (RS) 2b-4 (RST) 2b-7 (T10)
    moderate-moderate  a_s 0.08  a_t 0.24  alpha 0.6    0.32 / 38.20 slots   datasets 2b-2      2b-5       2b-8
    strong-strong      a_s 0.4   a_t 1     alpha 0.08   1.6  / infinite      datasets 2b-3      2b-6       2b-9

Leave-out schemes (10% testing values): RS, 10% of the sites with all their slots; RST, 10% of all the space-time
values; T10, all the sites at the last 10 slots.

How this replica differs from the original data (approximations):
  * One realization per setting is simulated by circulant embedding of C on a regular space-time grid: the sites
    are nodes (i + 0.5) / n of an n x n grid of [0, 1]^2 (default n = 400, spacing 0.0025) and the slots are
    0..99; the grid is embedded in a periodic Ls x Ls x Lt grid (spacing kept, so the periodic domain is
    [0, Ls/n)^2 x [0, Lt) slots) and the field is the real part of an FFT of complex white noise scaled by the
    square roots of the embedding eigenvalues (the eigenvalues are the 3D DCT-I of one octant of C).  Lt = 256
    makes all the lags 0..99 slots exact; the spatial period is large (default 4, 8 and 12 for the weak,
    moderate and strong settings) because the spatial range grows with the time lag (a_s psi(u)^(beta/2)) and
    the covariance truncated at half the period is otherwise markedly indefinite (period 2: negative mass
    1e-4 for moderate-moderate, 2e-2 for strong-strong).
  * The truncated covariance is not exactly positive definite on the periodic grid: the most negative eigenvalue
    (relative to the largest) and the fraction of negative mass (sum |lambda_-| / sum |lambda|) are reported and
    stored in 'meta' (default embeddings: weak none, moderate 1.7e-8, strong 1.3e-6, the most negative
    eigenvalue at most 4e-8 of the largest); the negative eigenvalues are clipped to zero, which adds a small positive-definite
    long-range perturbation to the covariance (the realized variance is sum max(lambda, 0) / N instead of
    sigma2, also reported).  The script stops when the negative mass exceeds --max-neg-frac.
  * The 10,000 sites are drawn uniformly at random among the n^2 grid nodes (fixed seed) instead of uniformly
    in [0, 1]^2, so the coordinates are quantized to the grid spacing; the same sites serve all settings.
  * The three datasets of a setting (RS, RST, T10) are three splits of the same realization.
  * --nuggets adds N(0, tau2) noise to all the observed values (training and testing) for each given tau2 > 0;
    tau2 = 0 (default) is the model of the paper.  The noise-free testing values are stored in 'mtest' (and the
    noise-free training values in 'mtrain' when tau2 > 0).

Output, one npz per dataset in $PSCRATCH/kaust2022/: kaust2022_2b<k>_replica.npz (tau2 = 0) or
kaust2022_2b<k>_replica_nug<tau2>.npz, with xtrain (N, 3) = (x, y, t_norm), t_norm = slot / 99, ytrain, xtest,
ytest, mtest, slot_train, slot_test, site_train, site_test (site ids 0..9999, sorted by grid index) and meta (a
json string with the parameters, the grid, the seeds and the embedding diagnostics).  The training values are
ordered by slot, then site; the testing values are randomly permuted (the drivers compute the predictive
variances on the first testing values).

Usage (on a CPU compute node; measured peak memory 48 GB for moderate, 108 GB for strong, < 1 minute each):
  python kaust2022_2b_replica.py moderate --nuggets 0,0.01
  python kaust2022_2b_replica.py weak
  python kaust2022_2b_replica.py strong
  python kaust2022_2b_replica.py moderate --check 800x256,3200x256    (embedding diagnostics only)
e.g. sbatch -A m2957 -q debug -C cpu -N 1 -t 00:30:00 --wrap "module load python; python kaust2022_2b_replica.py moderate"
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft
import scipy.special as sp

SIGMA2, BETA, NU = 0.9, 0.9, 1.0
SETTINGS = {  # name: covariance parameters, dataset numbers for (RS, RST, T10), seed offset, default Ls (n = 400)
    'weak': dict(a_s=0.02, a_t=1.0, alpha=0.6, datasets=(1, 4, 7), seed=1, Ls=1600),
    'moderate': dict(a_s=0.08, a_t=0.24, alpha=0.6, datasets=(2, 5, 8), seed=2, Ls=3200),
    'strong': dict(a_s=0.4, a_t=1.0, alpha=0.08, datasets=(3, 6, 9), seed=3, Ls=4800),
}
NSITES, NSLOTS, TEST_FRACTION = 10000, 100, 0.1
SEED = 2022
NTHREADS = os.cpu_count()


def matern(r, nu=NU):
    """M_nu(r) = r^nu K_nu(r) / (2^(nu-1) Gamma(nu)), M_nu(0) = 1."""
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    pos = r > 0
    rp = r[pos]
    if nu == 0.5:
        out[pos] = np.exp(-rp)
    elif nu == 1.0:
        out[pos] = rp * sp.k1(rp)
    else:
        out[pos] = rp ** nu * sp.kv(nu, rp) / (2 ** (nu - 1) * sp.gamma(nu))
    return out


def gneiting(h, u, a_s, a_t, alpha, sigma2=SIGMA2, beta=BETA, nu=NU):
    """C(h, u) for spatial distances h and time lags u (slots), broadcast."""
    psi = a_t * np.abs(u) ** (2 * alpha) + 1
    return sigma2 / psi * matern(h / (a_s * psi ** (beta / 2)), nu)


def effective_ranges(a_s, a_t, alpha, nu=NU):
    """Distances (space, at u = 0) and lags (time slots, at h = 0) where the correlation is 0.05."""
    from scipy.optimize import brentq
    space = a_s * brentq(lambda r: matern(np.array([r]), nu)[0] - 0.05, 1e-6, 100)
    time_ = (19 / a_t) ** (1 / (2 * alpha))  # 1 / psi(u) = 0.05
    return space, time_


def octant_covariance(p, Ls, Lt, dx):
    """C at lags (i dx, j dx, k), 0 <= i, j <= Ls/2, 0 <= k <= Lt/2: one octant of the (even) periodic base."""
    hx = dx * np.arange(Ls // 2 + 1)
    r = np.sqrt(hx[:, None] ** 2 + hx[None, :] ** 2)
    c = np.empty((Ls // 2 + 1, Ls // 2 + 1, Lt // 2 + 1))

    def slab(k):
        c[:, :, k] = gneiting(r, k, p['a_s'], p['a_t'], p['alpha'])
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(slab, range(Lt // 2 + 1)))
    return c


def multiplicity(L):
    """Number of full-grid frequencies represented by each octant index 0..L/2 along one axis."""
    m = np.full(L // 2 + 1, 2.0)
    m[0] = m[-1] = 1
    return m


def embedding_eigenvalues(p, Ls, Lt, dx):
    """Octant of the circulant-embedding eigenvalues (3D DCT-I of the octant covariance) and diagnostics."""
    t0 = time.time()
    lam = scipy.fft.dctn(octant_covariance(p, Ls, Lt, dx), type=1, workers=NTHREADS, overwrite_x=True)
    ms, mt = multiplicity(Ls), multiplicity(Lt)
    w = ms[:, None, None] * ms[None, :, None] * mt[None, None, :]
    N = Ls * Ls * Lt
    neg = lam < 0
    lam_max, lam_min = lam.max(), lam.min()
    neg_mass = float((w * np.where(neg, -lam, 0)).sum())
    pos_mass = float((w * np.where(neg, 0, lam)).sum())
    diag = dict(Ls=Ls, Lt=Lt, dx=dx, N=N, lambda_max=float(lam_max), lambda_min=float(lam_min),
                min_over_max=float(lam_min / lam_max), n_negative=int(w[neg].sum()),
                neg_mass_fraction=neg_mass / (neg_mass + pos_mass),
                variance_true=float(gneiting(0.0, 0.0, p['a_s'], p['a_t'], p['alpha'])),
                variance_from_trace=(pos_mass - neg_mass) / N, variance_after_clipping=pos_mass / N,
                seconds=time.time() - t0)
    return lam, diag


def print_diag(name, d):
    print("%s embedding %dx%dx%d (period %.2f x %.2f x %d slots): lambda_min/lambda_max %.3e, %d negative "
          "eigenvalues, negative mass fraction %.3e, variance %.4f -> %.4f after clipping (%.1f s)"
          % (name, d['Ls'], d['Ls'], d['Lt'], d['Ls'] * d['dx'], d['Ls'] * d['dx'], d['Lt'], d['min_over_max'],
             d['n_negative'], d['neg_mass_fraction'], d['variance_from_trace'], d['variance_after_clipping'],
             d['seconds']), flush=True)


def fill_normal(a, seed, nchunks=64):
    """Fills a (C-contiguous float64) with N(0, 1) values, in parallel chunks along axis 0 (reproducible)."""
    gens = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(nchunks)]
    bounds = np.linspace(0, a.shape[0], nchunks + 1).astype(int)

    def chunk(i):
        if bounds[i + 1] > bounds[i]:
            gens[i].standard_normal(out=a[bounds[i]:bounds[i + 1]])
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(chunk, range(nchunks)))


def simulate(lam, Ls, Lt, n, nslots, seed):
    """One realization on the n x n x nslots corner of the periodic grid from the octant eigenvalues."""
    N = Ls * Ls * Lt
    amp = np.sqrt(np.maximum(lam, 0) / N)
    ix = np.minimum(np.arange(Ls), Ls - np.arange(Ls))
    it = np.minimum(np.arange(Lt), Lt - np.arange(Lt))
    w = np.empty((Ls, Ls, Lt), dtype=np.complex128)
    fill_normal(w.view(np.float64), seed)

    def scale(i):
        w[i] *= amp[ix[i]][np.ix_(ix, it)]
    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(scale, range(Ls)))
    w = scipy.fft.fftn(w, workers=NTHREADS, overwrite_x=True)
    return np.ascontiguousarray(w[:n, :n, :nslots].real)


def splits(nsites, nslots, rng):
    """Boolean (nsites, nslots) testing masks of the RS, RST and T10 schemes."""
    rs = np.zeros((nsites, nslots), dtype=bool)
    rs[rng.choice(nsites, int(round(TEST_FRACTION * nsites)), replace=False)] = True
    rst = np.zeros(nsites * nslots, dtype=bool)
    rst[rng.choice(nsites * nslots, int(round(TEST_FRACTION * nsites * nslots)), replace=False)] = True
    t10 = np.zeros((nsites, nslots), dtype=bool)
    t10[:, nslots - int(round(TEST_FRACTION * nslots)):] = True
    return {'RS': rs, 'RST': rst.reshape(nsites, nslots), 'T10': t10}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('setting', choices=sorted(SETTINGS))
    ap.add_argument('--n', type=int, default=400, help='spatial grid nodes per direction in [0, 1] (spacing 1/n)')
    ap.add_argument('--Ls', type=int, default=0, help='periodic spatial grid size (default per setting for n = 400:'
                    ' weak 1600, moderate 3200, strong 4800, i.e. periods 4, 8, 12; else 2n)')
    ap.add_argument('--Lt', type=int, default=256, help='periodic time grid size (slots, >= 2 x 99)')
    ap.add_argument('--nuggets', default='0', help='comma-separated nugget variances; one set of files each')
    ap.add_argument('--max-neg-frac', type=float, default=1e-3, help='largest negative mass fraction accepted')
    ap.add_argument('--check', default='', help='only print the embedding diagnostics of these LsxLt sizes')
    ap.add_argument('--outdir', default=os.path.join(os.environ.get('PSCRATCH', '.'), 'kaust2022'))
    args = ap.parse_args()
    p = SETTINGS[args.setting]
    dx = 1.0 / args.n
    rs, rt = effective_ranges(p['a_s'], p['a_t'], p['alpha'])
    print("%s: a_s %g a_t %g alpha %g; effective ranges %.4f (space), %.2f slots (time); %d threads"
          % (args.setting, p['a_s'], p['a_t'], p['alpha'], rs, rt, NTHREADS), flush=True)
    if args.check:
        for size in args.check.split(','):
            Ls, Lt = map(int, size.split('x'))
            print_diag(args.setting, embedding_eigenvalues(p, Ls, Lt, dx)[1])
        return
    Ls = args.Ls or (p['Ls'] if args.n == 400 else 2 * args.n)
    assert Ls >= 2 * args.n - 1 and args.Lt >= 2 * (NSLOTS - 1) and Ls % 2 == 0 and args.Lt % 2 == 0
    lam, diag = embedding_eigenvalues(p, Ls, args.Lt, dx)
    print_diag(args.setting, diag)
    if diag['neg_mass_fraction'] > args.max_neg_frac:
        raise SystemExit("negative mass fraction %.3e > %.1e: embedding too small or not valid for %s"
                         % (diag['neg_mass_fraction'], args.max_neg_frac, args.setting))
    t0 = time.time()
    field = simulate(lam, Ls, args.Lt, args.n, NSLOTS, SEED * 10 + p['seed'])
    del lam
    print("simulated in %.1f s; grid field mean %.4f variance %.4f" % (time.time() - t0, field.mean(), field.var()), flush=True)

    # Sites (shared by all the settings), values, splits.
    rng = np.random.default_rng(SEED)
    node = np.sort(rng.choice(args.n * args.n, NSITES, replace=False))
    ixs, iys = np.divmod(node, args.n)
    sxy = np.stack([(ixs + 0.5) * dx, (iys + 0.5) * dx], axis=1)
    z = field[ixs, iys, :]  # (NSITES, NSLOTS)
    del field
    masks = splits(NSITES, NSLOTS, np.random.default_rng(SEED + 1))
    site = np.repeat(np.arange(NSITES)[:, None], NSLOTS, axis=1)
    slot = np.repeat(np.arange(NSLOTS)[None, :], NSITES, axis=0)
    os.makedirs(args.outdir, exist_ok=True)
    for tau2 in (float(v) for v in args.nuggets.split(',')):
        noise = (np.sqrt(tau2) * np.random.default_rng(SEED + 2).standard_normal(z.shape)) if tau2 > 0 else 0
        y = z + noise
        for scheme, k in zip(('RS', 'RST', 'T10'), p['datasets']):
            test = masks[scheme]
            # training ordered by slot then site; testing randomly permuted
            tr = np.flatnonzero(~test.T.ravel())
            tr_site, tr_slot = site.T.ravel()[tr], slot.T.ravel()[tr]
            te_site, te_slot = site[test], slot[test]
            perm = np.random.default_rng(SEED + 3 + k).permutation(len(te_site))
            te_site, te_slot = te_site[perm], te_slot[perm]
            X = lambda s, t: np.column_stack([sxy[s], t / (NSLOTS - 1.0)])
            meta = dict(description='replica of KAUST 2022 sub-competition 2b dataset %d (%s, %s), kaust2022_2b_replica.py'
                        % (k, args.setting, scheme), dataset=k, setting=args.setting, scheme=scheme,
                        covariance='C(h,u) = sigma2/psi(u) M_nu(||h||/(a_s psi(u)^(beta/2))), psi(u) = a_t|u|^(2 alpha)+1, u in slots',
                        sigma2=SIGMA2, beta=BETA, nu=NU, a_s=p['a_s'], a_t=p['a_t'], alpha=p['alpha'],
                        effective_range_space=rs, effective_range_time_slots=rt,
                        effective_range_time_tnorm=rt / (NSLOTS - 1), nugget=tau2,
                        nsites=NSITES, nslots=NSLOTS, t_norm='slot/%d' % (NSLOTS - 1),
                        grid_n=args.n, grid_spacing=dx, site_coordinates='((i+0.5)/n, (j+0.5)/n)',
                        embedding=diag, clipped_negative_eigenvalues=diag['n_negative'] > 0,
                        seeds=dict(sites=SEED, field=SEED * 10 + p['seed'], splits=SEED + 1, noise=SEED + 2,
                                   test_permutation=SEED + 3 + k),
                        field_sample_variance=float(z.var()), ntrain=len(tr), ntest=len(te_site))
            arrays = dict(xtrain=X(tr_site, tr_slot), ytrain=y[tr_site, tr_slot], xtest=X(te_site, te_slot),
                          ytest=y[te_site, te_slot], mtest=z[te_site, te_slot], slot_train=tr_slot, slot_test=te_slot,
                          site_train=tr_site, site_test=te_site, meta=json.dumps(meta))
            if tau2 > 0:
                arrays['mtrain'] = z[tr_site, tr_slot]
            name = 'kaust2022_2b%d_replica%s.npz' % (k, '' if tau2 == 0 else '_nug%g' % tau2)
            out = os.path.join(args.outdir, name)
            np.savez(out, **arrays)
            print("%s: %s %s, %d training and %d testing values, ytrain variance %.4f"
                  % (out, args.setting, scheme, len(tr), len(te_site), arrays['ytrain'].var()), flush=True)


if __name__ == '__main__':
    main()
