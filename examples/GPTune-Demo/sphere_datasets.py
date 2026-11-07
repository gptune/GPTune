#! /usr/bin/env python
"""
Datasets on the unit sphere for the GP comparisons (kaust_inla_wendland.py reads the npz files: xtrain, ytrain,
xtest, ytest, xtest2, ytest2, and mtest/mtest2 when the noise-free values are known). The locations are 3D
Cartesian points on the unit sphere, so an isotropic kernel of the 3D distance is a kernel of the chordal
distance, which is positive definite on the sphere (as the RBF and Matern kernels of george and butterflypack).

  modis   one day of Aqua MODIS daytime SST (L3 mapped, 4 km, NASA GSFC OBPG, R2019, ERDDAP dataset
          erdMH1sstd1dayR20190 at coastwatch.pfeg.noaa.gov), downloaded with a stride of 3 (0.125 degrees) by
              curl -o modis_sst_2021-07-01_stride3.nc "https://coastwatch.pfeg.noaa.gov/erddap/griddap/\
erdMH1sstd1dayR20190.nc?sstMasked[(2021-07-01T12:00:00Z)][(89.97916):3:(-89.97918)][(-179.9792):3:(179.9792)]"
          The valid (cloud-free, ocean) pixels are the data. A zonal trend (a degree-4 polynomial in
          sin(latitude), fitted to the training values) is removed: ytrain/ytest are the anomalies, and
          trend_coefficients rebuild the SST.
  grf     a Gaussian random field with an isotropic covariance of the chordal distance r, sampled exactly by
          spherical harmonic synthesis (ducc0), plus independent N(0, tau^2) noise, at the MODIS locations (the
          satellite design) or at uniformly random points:
              rbf       sigma^2 exp(-r^2 / (2 l^2))
              matern32  sigma^2 (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)
          mtest/mtest2 are the noise-free values.

Test sets (both datasets): testing set 1 is a random 10% of the locations; testing set 2 is all the locations
in 40 random 5 x 5 degree boxes, removed from the training data (gaps, as cloud cover).

Usage:
  python sphere_datasets.py modis [nc file] [out.npz]
  python sphere_datasets.py grf <rbf|matern32> <l> <sigma2> <tau2> [modis|uniform:<N>] [out.npz]
  python sphere_datasets.py lonlat <in.npz> [out.npz]   (the same data with the inputs (lon + 180) / 360 and
          (lat + 90) / 180: the planar longitude-latitude coordinates of a tensor-lattice INLA model, which
          neither wraps around in longitude nor accounts for the convergence of the meridians)
Defaults in $PSCRATCH/sphere; ducc0 (pip install --target $PSCRATCH/sphere/pylib ducc0) is needed for grf.
"""

import os
import sys

import numpy as np

BASE = os.path.join(os.environ.get('PSCRATCH', '.'), 'sphere')
SEED = 2026
NBOX, BOX_DEG = 40, 5.0


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def modis_locations(nc_file):
    import netCDF4
    d = netCDF4.Dataset(nc_file)
    sst = d.variables['sstMasked'][0]
    lat, lon = np.asarray(d.variables['latitude'][:]), np.asarray(d.variables['longitude'][:])
    values = np.ma.filled(sst.astype(float), np.nan)
    i, j = np.nonzero(np.isfinite(values))
    return lat[i], lon[j], values[i, j]


def split(lat, lon, rng):
    """Testing set 2: NBOX random BOX_DEG boxes (each with data); testing set 1: random 10% of the rest."""
    in_box = np.zeros(len(lat), dtype=bool)
    nbox = 0
    while nbox < NBOX:
        k = rng.integers(len(lat))  # box corners at data points: boxes lie over the ocean
        la0, lo0 = lat[k] - BOX_DEG / 2, lon[k] - BOX_DEG / 2
        box = (lat >= la0) & (lat < la0 + BOX_DEG) & (lon >= lo0) & (lon < lo0 + BOX_DEG)
        if box.sum() > 100 and not (box & in_box).any():
            in_box |= box
            nbox += 1
    rest = np.nonzero(~in_box)[0]
    test1 = rng.choice(rest, len(rest) // 10, replace=False)
    train = np.setdiff1d(rest, test1)
    test2 = rng.permutation(np.nonzero(in_box)[0])
    return train, test1, test2


_GAUSS = {}


def kernel_spectrum(kind, l, sigma2, lmax, nquad=None):
    """C_n, n <= lmax, of k(t) = sum_n (2n+1)/(4 pi) C_n P_n(t), t = cos(angle): C_n = 2 pi int k(t) P_n(t) dt."""
    from scipy.special import roots_legendre  # (tridiagonal: numpy's leggauss is O(n^3))
    nquad = nquad or max(4 * lmax, 8000)
    if nquad not in _GAUSS:
        _GAUSS[nquad] = roots_legendre(nquad)
    t, w = _GAUSS[nquad]
    r = np.sqrt(np.maximum(2 - 2 * t, 0))
    if kind == 'rbf':
        k = sigma2 * np.exp(-r ** 2 / (2 * l ** 2))
    elif kind == 'matern32':
        k = sigma2 * (1 + np.sqrt(3) * r / l) * np.exp(-np.sqrt(3) * r / l)
    else:
        raise ValueError(kind)
    c = np.empty(lmax + 1)
    p0, p1 = np.ones_like(t), t
    c[0] = 2 * np.pi * np.sum(w * k * p0)
    if lmax > 0:
        c[1] = 2 * np.pi * np.sum(w * k * p1)
    for n in range(2, lmax + 1):
        p0, p1 = p1, ((2 * n - 1) * t * p1 - (n - 1) * p0) / n
        c[n] = 2 * np.pi * np.sum(w * k * p1)
    return np.maximum(c, 0)


def grf(kind, l, sigma2, lat, lon, rng, tol=1e-5):
    """A sample of the field at (lat, lon) by spherical harmonic synthesis, with lmax keeping all but tol of sigma2."""
    sys.path.insert(0, os.path.join(BASE, 'pylib'))
    import ducc0
    lmax = 64
    while True:
        c = kernel_spectrum(kind, l, sigma2, lmax)
        captured = np.sum((2 * np.arange(lmax + 1) + 1) * c) / (4 * np.pi)
        if captured >= (1 - tol) * sigma2 or lmax >= 4096:
            break
        lmax *= 2
    print("%s field, l %g: lmax %d keeps %.6f of the variance" % (kind, l, lmax, captured / sigma2))
    alm = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=np.complex128)
    idx = 0
    for m in range(lmax + 1):
        n = lmax + 1 - m
        sd = np.sqrt(c[m:])
        if m == 0:
            alm[idx:idx + n] = sd * rng.standard_normal(n)
        else:
            alm[idx:idx + n] = sd * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        idx += n
    loc = np.stack([np.radians(90.0 - lat), np.radians(lon) % (2 * np.pi)], axis=1)
    f = ducc0.sht.synthesis_general(alm=alm[None, :], spin=0, lmax=lmax, loc=loc, epsilon=1e-10,
                                    nthreads=min(32, os.cpu_count()))[0]
    return np.asarray(f.real, dtype=float), lmax


def main():
    rng = np.random.default_rng(SEED)
    nc = os.path.join(BASE, 'modis_sst_2021-07-01_stride3.nc')
    if sys.argv[1] == 'modis':
        nc = sys.argv[2] if len(sys.argv) > 2 else nc
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, 'sphere_modis_sst_20210701.npz')
        lat, lon, sst = modis_locations(nc)
        train, test1, test2 = split(lat, lon, rng)
        z = np.sin(np.radians(lat))
        coeffs = np.polynomial.polynomial.polyfit(z[train], sst[train], 4)
        anomaly = sst - np.polynomial.polynomial.polyval(z, coeffs)
        xyz = to_xyz(lat, lon)
        np.savez(out, xtrain=xyz[train], ytrain=anomaly[train], xtest=xyz[test1], ytest=anomaly[test1],
                 xtest2=xyz[test2], ytest2=anomaly[test2], lat=lat, lon=lon, sst=sst, train=train, test1=test1,
                 test2=test2, trend_coefficients=coeffs)
        print("%s: %d training, %d + %d testing pixels; SST anomaly std %.3f (SST std %.3f)"
              % (out, len(train), len(test1), len(test2), anomaly[train].std(), sst[train].std()))
    elif sys.argv[1] == 'grf':
        kind, l, sigma2, tau2 = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        design = sys.argv[6] if len(sys.argv) > 6 else 'modis'
        if design == 'modis':
            lat, lon, _ = modis_locations(nc)
        else:
            n = int(design.split(':')[1])
            lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
            lon = rng.uniform(-180, 180, n)
        out = sys.argv[7] if len(sys.argv) > 7 else os.path.join(
            BASE, 'sphere_grf_%s_l%g_s%g_t%g_%s.npz' % (kind, l, sigma2, tau2, design.replace(':', '')))
        f, lmax = grf(kind, l, sigma2, lat, lon, rng)
        y = f + np.sqrt(tau2) * rng.standard_normal(len(f))
        train, test1, test2 = split(lat, lon, rng)
        xyz = to_xyz(lat, lon)
        np.savez(out, xtrain=xyz[train], ytrain=y[train], xtest=xyz[test1], ytest=y[test1], mtest=f[test1],
                 xtest2=xyz[test2], ytest2=y[test2], mtest2=f[test2], lat=lat, lon=lon, train=train, test1=test1,
                 test2=test2, truth=np.array([sigma2, l, tau2]), lmax=lmax)
        print("%s: %d training, %d + %d testing values; field variance %.3f (target %g)"
              % (out, len(train), len(test1), len(test2), f.var(), sigma2))
    elif sys.argv[1] == 'lonlat':
        src = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else src.replace('.npz', '_lonlat.npz')
        z = dict(np.load(src))
        # the ERDDAP coordinates are within ~1e-4 degree of the 0.125 degree grid: snap them to it, so that the
        # inputs are exactly on a lattice (the INLA lattice is then the data grid)
        step = 0.125
        lon = z['lon'].min() + np.round((z['lon'] - z['lon'].min()) / step) * step
        lat = z['lat'].max() - np.round((z['lat'].max() - z['lat']) / step) * step
        print("snapped by at most %.2e degree" % max(np.abs(lon - z['lon']).max(), np.abs(lat - z['lat']).max()))
        ll = np.stack([(lon + 180) / 360, (lat + 90) / 180], axis=1)
        for key, idx in (('xtrain', 'train'), ('xtest', 'test1'), ('xtest2', 'test2')):
            z[key] = ll[z[idx]]
        np.savez(out, **z)
        print("%s: longitude-latitude inputs of %s" % (out, src))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
