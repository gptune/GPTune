# ButterflyPACK H2 (GPU): negative determinant sign for an SPD matrix

A standalone reproducer: numpy plus ButterflyPACK's own Python interface (`dPy_BPACK_wrapper.py` and
`dPy_BPACK_worker.py` from `build_gpu/lib`). No GPTune or george. It runs in about one minute on 4 GPU nodes.

## The bug

For the SPD matrix

    K(i, j) = exp(-||x_i - x_j|| / 0.01) + 0.05 [i == j]

on n 2D points, ButterflyPACK's H2 factorization (format 7, unstructured tree, GPU) reports determinant
sign **−1** at n = 100K and 400K, and +1 at 20K and 200K. The exponential (Matérn-1/2) kernel is positive
definite in any dimension, so K ≥ 0.05·I and det K > 0: the sign must be +1.

- The wrong sign already appears at factor time, in the worker log line `logdet: -1 ...`.
  `c_bpack_logdet` (`bpack_logdet` in Python) returns the same sign.
- The magnitude log|det K| is right. At 100K it is identical to the GPTune runs where this was found.
- Solves are right: residuals are about 1e-13.

So the sign bookkeeping is wrong, not the factors. Suspects: the parity of the LU pivots
(`accumulate_logdet_lu`, and the device sums in `device_logdet_kept_level`), or how the per-level or
per-rank signs are combined. The two failing sizes have different numbers of levels (8 and 9), and so do
the two passing ones (7 and 8).

## Run

```
cd examples/GPTune-Demo/spatial_h2/logdet_sign_repro
SIZES=20000,100000,200000,400000 sbatch run_logdet_sign_repro.sbatch
```

The sbatch uses premium on account `m4293_g`. Settings come from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SIZES` | `20000,100000` | numbers of points, comma separated |
| `NMPI` | 16 | MPI ranks, one per A100 |
| `RANKS_PER_NODE` | 4 | |
| `NMIN_LEAF` | 8 | |
| `REDUCTION_THRESHOLD` | 4 | |
| `VERBOSITY` | 1 | needed for the factor-time `logdet:` line |
| `BPACK_DIR` | this checkout (`examples/ButterflyPACK/ButterflyPACK`) | its `build_gpu/lib` is used |

Worker options, as in the runs where the bug was found:

```
--xyzsort 0 --format 7 --sym 1 --reduction_threshold 4 --tol_comp 1e-10 --h2_id_proxy 2 --baca_batch 32
--h2_id_radius 2 --nmin_leaf 8 --errsol 0 --verbosity 1 --h2_unstructured 1 --H2_use_gpu 1
--H2_XRR_factor 1 --h2_lazy_schur 2 --h2_use_sketch 2
```

The runtime environment is the one of the GPTune H2 runs:
- modules: python/3.13, PrgEnv-gnu, gcc-native/14, cray-libsci/26.03.0, cray-mpich/9.1.0, cudatoolkit,
  craype-accel-nvidia80
- mpi4py from the user site (`$PYTHONUSERBASE`)
- the Cray GTL and a sequential OpenBLAS preloaded in the workers

## Files

| File | Contents |
|---|---|
| `points.npy` | 900K × 2 locations of the KAUST 2023 dataset 2 (94% of the points in the bottom-left quarter of [0,1]²). Size n uses `np.random.default_rng(0).choice(900000, n, replace=False)`. Not in git (14 MB): `np.save("points.npy", np.load("$PSCRATCH/kaust2023/kaust2023_2b_dataset02.npz")["xtrain"])`. |
| `logdet_sign_repro.py` | Driver: per size, `bpack_factor`, `bpack_logdet`, one solve with its residual (K applied with `bpack_mult`), and up to 20K points the exact log det K from a dense Cholesky factorization. The GPU entry evaluator is CUDA source text compiled with NVRTC. |
| `repro_block_funcs.py` | The same entries on the host. |
| `run_logdet_sign_repro.sbatch` | Starts the workers and the driver; prints the factor-time `logdet:` lines at the end. |

## Output of job 59969944 (ButterflyPACK 32bacba)

```
RESULT n 20000: bpack_logdet sign +1, log|det| -2.3372267858e+04 | solve residual 1.5e-14 | dense log det K -2.3372235520e+04 (H2 log|det| - dense -3.23e-02) | 18.3 s | ok
RESULT n 100000: bpack_logdet sign -1, log|det| -1.6500549109e+05 | solve residual 5.3e-14 | 1.8 s | WRONG SIGN (K is SPD)
RESULT n 200000: bpack_logdet sign +1, log|det| -3.6801128413e+05 | solve residual 9.1e-14 | 2.4 s | ok
RESULT n 400000: bpack_logdet sign -1, log|det| -8.0578609890e+05 | solve residual 1.3e-13 | 4.0 s | WRONG SIGN (K is SPD)
factor-time logdet lines of the workers:
logdet: 1 -23372.267858170679
logdet: -1.00000000000000000e+00 -1.65005491087510163e+05
logdet: 1.00000000000000000e+00 -3.68011284131140681e+05
logdet: -1.00000000000000000e+00 -8.05786098895497387e+05
```

The tree for each size (16 ranks):

| n | Levels | Occupied leaves | Points per occupied leaf | Sign |
|---|---|---|---|---|
| 20K | 7 (6 from N alone) | 2,037 | 9.8 | +1 |
| 100K | 8 (7 from N alone) | 8,986 | 11.1 | **−1** |
| 200K | 8 (as from N) | | | +1 |
| 400K | 9 (8 from N alone) | 35,942 | 11.1 | **−1** |

## A separate issue the same run shows: compression accuracy

The quick check reports `acc_mvp` well above `tol_comp` 1e-10 at two of the sizes:

| n | `acc_mvp` | `acc_forward` | Note |
|---|---|---|---|
| 20K | 3.0e-8 | 2.0e-7 | log|det| differs from the dense value by 0.032 (1.4e-6 relative) |
| 100K | 3.2e-10 | 1.5e-8 | |
| 200K | 1.4e-10 | 1.4e-8 | |
| 400K | 6.0e-5 | 6.6e-3 | |

At 400K this error is large enough to make the analytic GP gradient disagree with central differences of
the likelihood. More context on both issues, and on the GPTune runs that hit them, is in
`../near_block_too_wide.md`, section "Root cause of the 'gradient error'".
