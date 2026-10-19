"""Host entries of the matrix of logdet_sign_repro.py, for the ButterflyPACK Python workers:

    K(i, j) = amplitude * exp(-sqrt(sum_d (x_id - x_jd)^2 / ell^2)) + yerr2_i [i == j]

with i, j 0-based global ids into meta["coordinates"]. The same formula as the GPU entry in
logdet_sign_repro.py (and george's exponential kernel).
"""
import numpy as np


def compute_block(rows, cols, meta):
    x = meta["coordinates"]
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    d = x[rows][:, None, :] - x[cols][None, :, :]
    q = np.einsum("ijk,ijk->ij", d, d) / meta["ell"] ** 2
    out = np.asfortranarray(meta["amplitude"] * np.exp(-np.sqrt(q)))
    same = rows[:, None] == cols[None, :]
    out[same] += np.broadcast_to(meta["yerr2"][rows][:, None], same.shape)[same]
    return out
