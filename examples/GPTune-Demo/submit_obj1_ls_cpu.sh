#!/bin/bash
# Objective 1 (the oscillatory 1D demo function) on CPU nodes with the length scale restricted to
# [0.02, 0.03], the range the unrestricted runs converged to when they converged at all.  The
# earlier campaign let l run over the GPTune default [1.01e-5, 2.718]; several runs stalled far
# from the optimum, and the 1.6M ones ended at l ~ 2e-4 where HODLR's ranks peak, so their timings
# measured the wrong hyperparameters.  HODLR at every size, H2 at 1.6M (the size where it stalled).
#
# All four optimizers run in one job, L-BFGS first, because MCMC and MALA take their time budget
# from the L-BFGS training files written beside them.
#
# These runs go in their own directories: a figure holds one objective under one set of settings,
# and mixing them with the unrestricted runs of scal_obj1_* would do neither.
#
# Rerun whenever slots free up; a row is skipped once submitted.  MAXJOBS (default 4 of the 5 the
# QOS allows) leaves a slot for other sessions.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
LS="0.02 0.03 0.025"   # minimum, maximum, initial length scale (linear scale)
OPT="gradient;finite difference;mcmc;mala"
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id runname NS fmt hours
    local id=$1 rn=$2 ns=$3 fmt=$4 hours=$5
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=$fmt NMPI=64 NTH=8 KNN=0 "OPTIMIZER=$OPT" "LENGTHSCALE=$LS" \
        RUN_NAME=$rn \
        sbatch -C cpu -N 4 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                   run dir              NS        fmt hours
try hodlrls_o1_N100k     scal_obj1ls_hodlrcpu 100001    1   04
try hodlrls_o1_N200k     scal_obj1ls_hodlrcpu 200001    1   06
try hodlrls_o1_N400k     scal_obj1ls_hodlrcpu 400001    1   08
try hodlrls_o1_N800k     scal_obj1ls_hodlrcpu 800001    1   10
try hodlrls_o1_N1600k    scal_obj1ls_hodlrcpu 1600001   1   12
try h2ls_o1_N1600k       scal_obj1ls_h2cpu    1600001   7   08
