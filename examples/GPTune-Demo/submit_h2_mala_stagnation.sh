#!/bin/bash
# H2 MALA without the 3x L-BFGS time cap, stopped by GPTune's stagnation test (20 Nov 2026): does the stop
# that ends the Wendland and INLA MALA runs at the L-BFGS optimum (Tables III and IV) also work for H2?
# 2D anisotropic kernel, objective 4, N = 100k, scaled-coordinate tree: the layout and settings of the
# row h2_2da_s_N100k of submit_aniso_scaled.sh (Table IX, job 61605206: 1 node, 4 ranks, nmin_leaf 64,
# region imbalance 1.15) and its ButterflyPACK build (the main checkout's build_gpu, eef6ed0 of 4 Nov).
# Unlike the sparse runs, the chain-agreement test stays on for MALA (GP_MCMC_AGREEMENT_TOL=5): an inexact
# H2 solve can give a spurious best sample, which only the agreement test keeps the sampling from stopping
# on (2D anisotropic 1.6M MCMC). The run is deterministic, so a replay of its chains (mcmc_chains_*.npz)
# gives the stop without the agreement test as well. 5 h sampling safety stop in a 6 h job.
#   bash submit_h2_mala_stagnation.sh
cd "$(dirname "$0")"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
RN=scal_h2gpu_2d_aniso_scaled_mala_stagnation
mkdir -p h2_unstructured_cmp/$RN
export BPACK_DIR=$PWD/../ButterflyPACK/ButterflyPACK
env USE_GPU=1 FORMAT=7 NMPI=4 RANKS_PER_NODE=4 NMIN_LEAF=64 H2_REGION_IMBALANCE=1.15 OPTIMIZER=mala \
    "NOISEVAR=1e-3 1e-1 1e-2" "LENGTHSCALE=1e-2 2.718281828 0.1" RUN_NAME=$RN \
    GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=100000 GP_MCMC_MAXTIME=18000 GP_MCMC_CONVERGENCE=stagnation GP_MCMC_AGREEMENT_TOL=5 \
    sbatch --parsable -A m5449_g -C gpu --gpus-per-node=4 -N 1 -q premium -t 06:00:00 --job-name=h2-mala-stag \
    --output=$PWD/h2_unstructured_cmp/$RN/GPTune_H2_unstr_%j.out \
    test_GP_h2_unstructured.sh aniso_scaled 100001 4 | tee -a h2_unstructured_cmp/$RN/submitted_jobs.txt
