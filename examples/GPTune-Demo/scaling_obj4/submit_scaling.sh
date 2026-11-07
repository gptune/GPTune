#!/bin/bash
# Scaling test of the sparse-GP hyperparameter training (objective 4, 2D Schwefel), GPU premium, 4 nodes,
# 16 SuperLU ranks, as two jobs (gpu_premium allows 5 submitted jobs per user): N = 100k to 800k in one,
# N = 1.6M in the other. See run_scaling_gpu.sbatch; plots: python ../plot_training_results.py . 4
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/scaling_obj4
for NS in "100000 200000 400000 800000" "1600000"; do
    SCALING_NS="$NS" RESULTS_DIR=$R sbatch --parsable -D $R --job-name=gp-scaling-N${NS// /-} $D/run_scaling_gpu.sbatch | sed "s/^/N=$NS: job /"
done
