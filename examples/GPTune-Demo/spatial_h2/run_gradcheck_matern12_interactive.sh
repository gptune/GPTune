#!/bin/bash
# GPU test of george's Matern1Kernel and Matern2Kernel (nu = 1 and 2, added 26 Nov 2026) in butterflypack's H2
# format: h2_gradient_check.py (the H2 log-likelihood and gradient against george's exact dense ones and central
# differences) at 5,000 and 20,000 points, for each kernel on a 2D field (the rough Matern-1/2 field at the KAUST
# 2023 dataset-2 design, 4 ranks) and on the 3D synthetic Matern-3/2 field on the sphere (1 rank), one after the
# other in a 1-node GPU interactive allocation (salloc runs its command on the login node, so the checks run in
# an interactive step, see run_h2_interactive.sh). Results in spatial_h2/gradcheck_matern12/<kernel>_<data>/.
# Usage: run_gradcheck_matern12_interactive.sh [account]
set -e
DEMO_DIR=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
cd $DEMO_DIR
ACCOUNT=${1:-m4293_g}
LOG=$DEMO_DIR/spatial_h2/interactive_reruns.log
OUT=$DEMO_DIR/spatial_h2/gradcheck_matern12
mkdir -p $OUT
cat > $OUT/runs.sh <<EOF
set -x
cd $DEMO_DIR
export DRIVER=gradcheck GP_DATASET=gradcheck_matern12 GC_SIZES=5000,20000 GC_DENSE_MAX=20000 GC_AMPLITUDE=1 H2_UNSTRUCTURED=1
export BPACK_DIR=$DEMO_DIR/../ButterflyPACK/ButterflyPACK_1d3d6d1
for kern in Matern1 Matern2; do
    for data in 2d sphere; do
        if [ \$data = 2d ]; then
            export GC_DATA=\$PSCRATCH/kaust2023/rough_ds2_nu0.5_r0.02.npz NMPI=4 NMIN_LEAF=16 REDUCTION_THRESHOLD=4 GC_LENGTHSCALE=0.02 GC_NOISE=0.05
        else
            export GC_DATA=\$PSCRATCH/sphere/sphere_grf_matern32_l0.05_s1_t0.01_uniform320000.npz NMPI=1 NMIN_LEAF=8 REDUCTION_THRESHOLD=8 GC_LENGTHSCALE=0.05 GC_NOISE=0.01
        fi
        export GC_KERN=\$kern RESULTS_DIR=$OUT/\${kern}_\${data}
        echo "\$(date '+%F %T') gradcheck \$kern \$data: job \$SLURM_JOB_ID" >> $LOG
        bash run_spatial_h2_gpu.sbatch > $OUT/\${kern}_\${data}.out 2>&1 || echo "\$(date '+%F %T') gradcheck \$kern \$data failed" >> $LOG
    done
done
echo "\$(date '+%F %T') gradcheck matern12: job \$SLURM_JOB_ID done" >> $LOG
EOF
echo "$(date '+%F %T') gradcheck matern12: account $ACCOUNT, 1 node, BPACK 1d3d6d1, sizes 5000,20000" >> $LOG
salloc -A $ACCOUNT -q interactive -C gpu -N 1 --gpus-per-node=4 -t 01:00:00 -J gp-gradcheck \
    srun --interactive --preserve-env -N 1 -n 1 bash $OUT/runs.sh
