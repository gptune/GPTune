#!/bin/bash
# The original HODLR-vs-H2 scaling comparison, reproduced: objective 1 (the oscillatory 1D demo
# function of model_comparison_updated_bpack.py), both formats on CPU nodes, 64 MPI ranks with 8
# OpenMP threads each on 4 nodes, gradient and finite-difference L-BFGS and MCMC, as
# test_GP_hodlr_vs_bpack.sh ran them.  Options follow that script (H2: reduction_threshold 4,
# tol_comp 1e-11, h2_id_proxy 0; HODLR: xyzsort 1, IR_HODLR 10, tol_comp 1e-10, jitter_factor 0,
# lrlevel 0, reclr_leaf 5, baca_batch 16, nmin_leaf 128, knn 0), so that the figure can be compared
# with the earlier one.  knn 0 is the original setting, not the knn 20 the later campaigns use.
#
# Rerun whenever slots free up; a row is skipped once submitted.  MAXJOBS (default 4 of the 5 the
# QOS allows) leaves a slot for other sessions.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id runname NS fmt hours
    local id=$1 rn=$2 ns=$3 fmt=$4 hours=$5
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=$fmt NMPI=64 NTH=8 KNN=0 "OPTIMIZER=gradient;finite difference;mcmc" \
        RUN_NAME=$rn \
        sbatch -C cpu -N 4 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                 run dir            NS        fmt hours
try h2cpu_o1_N100k     scal_obj1_h2cpu    100001    7   04
try hodlrcpu_o1_N100k  scal_obj1_hodlrcpu 100001    1   04
try h2cpu_o1_N200k     scal_obj1_h2cpu    200001    7   06
try hodlrcpu_o1_N200k  scal_obj1_hodlrcpu 200001    1   06
try h2cpu_o1_N400k     scal_obj1_h2cpu    400001    7   08
try hodlrcpu_o1_N400k  scal_obj1_hodlrcpu 400001    1   08
try h2cpu_o1_N800k     scal_obj1_h2cpu    800001    7   10
try hodlrcpu_o1_N800k  scal_obj1_hodlrcpu 800001    1   10
try h2cpu_o1_N1600k    scal_obj1_h2cpu    1600001   7   12
try hodlrcpu_o1_N1600k scal_obj1_hodlrcpu 1600001   1   12
