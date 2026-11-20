#!/bin/bash
# Test of the MCMC/MALA convergence fixes of 20 Nov 2026 on the Wendland GP of the paper's Table III (tab:wend_opt):
# the sampling stops by GPTune's new default test (model_mcmc_convergence='stagnation': the best sample improved by
# < 1 nat over the last 50 steps and every chain's best is within 5 nats of it), with no L-BFGS reference, and MALA
# uses the preconditioner with the capped bound-curvature term (MCMCSampler_MALA.gradient_curvature). 6 h safety
# stop (GP_MCMC_MAXTIME). Compare with scaling_obj4_mala_uncapped/timeout_6h_61634303 (100k MALA, old code, split
# R-hat, killed at step 700) and scaling_obj4_mala_target (stopped within 2% of gradient L-BFGS).
# Usage: bash submit_stagnation.sh [optimizer:N ...]   (default mala:100000 mcmc:100000 mala:200000)
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/scaling_obj4_stagnation
for run in ${@:-mala:100000 mcmc:100000 mala:200000}; do
    O=${run%%:*}; N=${run##*:}
    GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=100000 GP_MCMC_MAXTIME=21600 GP_MCMC_CONVERGENCE=stagnation \
    SCALING_NS=$N SCALING_OPTIMIZERS=$O RESULTS_DIR=$R \
        sbatch --parsable -A m5449_g -q premium -t 07:30:00 -D $R --job-name=gp-$O-stag-N$N $D/run_scaling_gpu.sbatch | sed "s/^/$O N=$N: job /"
done
