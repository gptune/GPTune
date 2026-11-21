#!/bin/bash
# INLA MALA curve of the paper's Fig. 4 (fig:history), 2D Schwefel objective 4, N = 102,400, 16 GPUs, without
# the 3x L-BFGS time cap of the original run (inla_vs_wendland/59033221, capped at 169 s: 18 evaluations, no
# improvement on the initial guess). Stops by GPTune's default test since 20 Nov 2026 (model_mcmc_convergence=
# 'stagnation': the best sample has improved by < 1 nat over the last 50 steps; no chain-agreement test for
# MALA), as the Wendland MALA reruns of Table III (scaling_obj4_mala_stagnation); 6 h safety stop
# (GP_MCMC_MAXTIME). 40 history entries replayed for the test metrics (GP_REPLAY_POINTS), as the Wendland curve.
D=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
R=$D/inla_vs_wendland/mala_stagnation
GP_NS=102400 GP_MODELS=inla GP_OPTIMIZERS=mala GP_MCMC_UNCAPPED=1 GP_MCMC_MAXITER=100000 GP_MCMC_MAXTIME=21600 \
GP_MCMC_CONVERGENCE=stagnation GP_REPLAY_POINTS=40 RESULTS_DIR=$R \
    sbatch --parsable -A m5449_g -q premium -t 07:30:00 -D $R --job-name=gp-inla-mala-stag \
    --output=$R/inla_wendland_gpu_%j.out $D/run_inla_wendland_gpu.sbatch | sed "s/^/inla mala N=102400: job /"
