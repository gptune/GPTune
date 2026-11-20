#!/bin/bash
# MALA for the 100k, 200k and 400k rows of the paper's Wendland table (tab:wend_opt), replacing the runs capped at
# 3x the L-BFGS time (27 Sep, before pattern reuse): the sampling stops once the best sample's -loglikelihood is
# within 2% of that of gradient L-BFGS at the same N (../scaling_obj4_reuse/run_stats_obj4_N<N>_gradient.json,
# GP_MCMC_TARGET_RTOL), or after 6 h of sampling (GP_MCMC_MAXTIME), and then writes the test predictions (RMSE, CRPS)
# at the best sample. R-hat is printed every 100 steps but not used to stop. Current code (OpenMP K assembly,
# SuperLU pattern reuse), the same as the L-BFGS rows; repo GPTune (GP_MCMC_UNCAPPED=1 sets GPTUNE_FROM_REPO in
# superlu_gpu_env.sh). One 4-node job per N, 7.5 h wall time. Usage: bash submit_mala_target.sh [N ...]
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/scaling_obj4_mala_target
for N in ${@:-100000 200000 400000}; do
    GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=100000 GP_MCMC_MAXTIME=21600 \
    GP_MCMC_TARGET_RUN=$D/scaling_obj4_reuse/run_stats_obj4_N${N}_gradient.json GP_MCMC_TARGET_RTOL=0.02 \
    SCALING_NS=$N SCALING_OPTIMIZERS=mala RESULTS_DIR=$R \
        sbatch --parsable -A m5449_g -q premium -t 07:30:00 -D $R --job-name=gp-mala-N$N $D/run_scaling_gpu.sbatch | sed "s/^/N=$N: job /"
done
