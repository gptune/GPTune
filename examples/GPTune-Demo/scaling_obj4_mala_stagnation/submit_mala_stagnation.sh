#!/bin/bash
# MALA rows of the paper's Table III (tab:wend_opt), Wendland GP, 2D Schwefel objective 4, 16 GPUs, with the
# convergence test GPTune uses by default since 20 Nov 2026 and no L-BFGS reference: stop once the best sample
# has improved by < 1 nat over the last 50 steps (model_mcmc_convergence='stagnation'; for MALA without the
# chain-agreement test, model_mcmc_agreement_tol='auto'), MALA preconditioner with the capped bound-curvature
# term; 6 h safety stop (GP_MCMC_MAXTIME). The validation runs in scaling_obj4_stagnation still required every
# chain's best within 5 nats, which made MALA wait 2.6-6x the time it needed to find the optimum.
# The 100k run replays 40 history entries for the test metrics (GP_REPLAY_POINTS), so that it can also give the
# Wendland MALA curve of Fig. 4 (fig:history), whose MCMC curve has 40.
# Usage: bash submit_mala_stagnation.sh [N[:replay] ...]   (default 100000:40 400000)
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/scaling_obj4_mala_stagnation
for run in ${@:-100000:40 400000}; do
    N=${run%%:*}; P=20; [ "$run" != "$N" ] && P=${run##*:}
    GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=100000 GP_MCMC_MAXTIME=21600 GP_MCMC_CONVERGENCE=stagnation GP_REPLAY_POINTS=$P \
    SCALING_NS=$N SCALING_OPTIMIZERS=mala RESULTS_DIR=$R \
        sbatch --parsable -A m5449_g -q premium -t 07:30:00 -D $R --job-name=gp-mala-stag-N$N $D/run_scaling_gpu.sbatch | sed "s/^/mala N=$N replay=$P: job /"
done
