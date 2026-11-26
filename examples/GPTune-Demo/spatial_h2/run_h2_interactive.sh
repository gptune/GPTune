#!/bin/bash
# One H2 run of kaust_inla_wendland.py (run_spatial_h2_gpu.sbatch) in a 2-node GPU interactive allocation.
# Usage: run_h2_interactive.sh <sst_ea|grf_rbf|grf_matern32> [account]
#        KERN=<Exp|Matern32|Matern52> run_h2_interactive.sh ...   (the H2 kernel; default RBF, as in Table VI)
# salloc runs its command on the login node here (LaunchParameters use_interactive_step applies only to salloc
# without a command), so the run script is started in an interactive step (srun --interactive, which holds no
# resources) on the first node: the driver and worker rank 0 share that node's /dev/shm, and the workers' srun
# inside the script gets the whole allocation.
# The Table VI H2 reruns of 26 Nov 2026: the settings of the original runs (59932993, 59293492, 59295104) with
# the ButterflyPACK build 1d3d6d1 (absolute ID floor, H2_s_op) and the unscaled objective (norm=0), as the
# full-grid run 59985120.
set -e
DEMO_DIR=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
cd $DEMO_DIR
RUN=$1
ACCOUNT=${2:-m4293_g}
export DRIVER=kaust KAUST_VAR_POINTS=5000 GP_OPTIMIZERS=gradient H2_UNSTRUCTURED=1 REDUCTION_THRESHOLD=8 NMPI=8
export GP_MODELS="h2:iso=1:norm=0${KERN:+:kern=$KERN}"
export BPACK_DIR=$DEMO_DIR/../ButterflyPACK/ButterflyPACK_1d3d6d1
case $RUN in
    sst_ea)
        export GP_DATASET=sphere_modis_sst_ea KAUST_DATA=$PSCRATCH/sphere/sphere_modis_sst_20210701_equalarea.npz
        export NMIN_LEAF=32 GP_NOISE='1e-6 1 0.0883' GP_AMPLITUDE='1e-2 1e2 8.609' GP_LENGTHSCALE='2e-3 1 0.0504' ;;
    grf_rbf)
        export GP_DATASET=sphere_grf_rbf KAUST_DATA=$PSCRATCH/sphere/sphere_grf_rbf_l0.05_s1_t0.01_uniform320000.npz
        export NMIN_LEAF=8 GP_LENGTHSCALE='2e-3 1 0.1' ;;
    grf_matern32)
        export GP_DATASET=sphere_grf_matern32 KAUST_DATA=$PSCRATCH/sphere/sphere_grf_matern32_l0.05_s1_t0.01_uniform320000.npz
        export NMIN_LEAF=8 GP_LENGTHSCALE='2e-3 1 0.1' ;;
    *) echo "unknown run $RUN" >&2; exit 1 ;;
esac
LOG=$DEMO_DIR/spatial_h2/interactive_reruns.log
echo "$(date '+%F %T') $RUN: account $ACCOUNT, GP_MODELS=$GP_MODELS GP_DATASET=$GP_DATASET KAUST_DATA=$KAUST_DATA NMIN_LEAF=$NMIN_LEAF GP_NOISE=${GP_NOISE:-} GP_AMPLITUDE=${GP_AMPLITUDE:-} GP_LENGTHSCALE=$GP_LENGTHSCALE BPACK_DIR=$BPACK_DIR" >> $LOG
salloc -A $ACCOUNT -q interactive -C 'gpu&hbm80g' -N 2 --gpus-per-node=4 -t 02:00:00 -J gp-spatial-h2 \
    srun --interactive --preserve-env -N 1 -n 1 bash -c \
    'echo "$(date "+%F %T") '$RUN': job $SLURM_JOB_ID, interactive step on $(hostname)" >> '$LOG'; bash run_spatial_h2_gpu.sbatch > spatial_h2_gpu_${SLURM_JOB_ID}.out 2>&1; echo "$(date "+%F %T") '$RUN': job $SLURM_JOB_ID done, exit $?" >> '$LOG
