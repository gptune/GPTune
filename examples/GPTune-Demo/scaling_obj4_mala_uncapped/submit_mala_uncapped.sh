#!/bin/bash
# MALA for the 100k, 200k and 400k rows of the paper's Wendland table (tab:wend_opt), without the
# sampling-time cap of scaling_obj4 (3x the shorter L-BFGS time): up to 1000 steps per chain, stopping
# once the split Gelman-Rubin statistic over the second half of the chains is below 1.1 (the first 100k
# job, 61630349, used the plain statistic below 1.5 and stopped with one chain still descending; its files
# are in stopped_unsplit_rhat_61630349/). Usage: bash submit_mala_uncapped.sh [N:hours ...] Current code (OpenMP K assembly,
# SuperLU pattern reuse), the same as the L-BFGS rows in ../scaling_obj4_reuse; one 4-node job per N.
# The 6 h 100k job 61634303 hit its wall time at step ~700 without writing results (timeout_6h_61634303/), so the
# sampling now has a safety stop 40 min before the wall time (GP_MCMC_MAXTIME), after which the run writes its
# results and the R-hat at the stop. 200k and 400k were dropped (20 Nov 2026; cancelled_200k_61630353/).
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/scaling_obj4_mala_uncapped
for spec in ${@:-100000:12}; do
    N=${spec%%:*}; H=${spec##*:}
    GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=1000 GP_MCMC_MAXTIME=$((10#$H * 3600 - 2400)) SCALING_NS=$N SCALING_OPTIMIZERS=mala RESULTS_DIR=$R \
        sbatch --parsable -A m5449_g -q premium -t $H:00:00 -D $R --job-name=gp-mala-N$N $D/run_scaling_gpu.sbatch | sed "s/^/N=$N: job /"
done
