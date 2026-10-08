#!/bin/bash
# Objective 1 (the oscillatory 1D demo function) on GPU nodes, the counterpart of the CPU campaign
# in submit_obj1_ls_cpu.sh: the length scale restricted to [0.02, 0.03], all four optimizers,
# HODLR and H2 at every size.  There were no GPU runs of this objective at all before this.
#
# Layout: 8 ranks, 4 per node over 2 nodes, so every A100 of both nodes is used, one per rank,
# with 16 threads each.  (The 1D Schwefel GPU campaign used 2 ranks per node and so left half the
# GPUs of each node idle.)  8 ranks is a power of two, which is what 1D data requires.  KNN=0
# matches the CPU runs of this objective rather than the knn 20 of the later campaigns, so that
# the CPU and GPU figures of objective 1 differ only in the device.
#
# H2 on GPU necessarily differs from H2 on CPU in two settings the launcher chooses by device:
# tol_comp 1e-10 and h2_id_proxy 2 (CPU uses 1e-11 and 0), plus the box-path flags without which
# the GPU build silently runs on the host.
#
# Results go to their own directories, so the GPU figure is separate from the CPU one.
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
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=$fmt NMPI=8 RANKS_PER_NODE=4 KNN=0 "OPTIMIZER=$OPT" "LENGTHSCALE=$LS" \
        RUN_NAME=$rn \
        sbatch -C gpu --gpus-per-node=4 -N 2 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                     run dir               NS       fmt hours
try h2gpu_o1ls_N100k       scal_obj1ls_h2gpu     100001    7   03
try hodlrgpu_o1ls_N100k    scal_obj1ls_hodlrgpu  100001    1   03
try h2gpu_o1ls_N200k       scal_obj1ls_h2gpu     200001    7   03
try hodlrgpu_o1ls_N200k    scal_obj1ls_hodlrgpu  200001    1   03
try h2gpu_o1ls_N400k       scal_obj1ls_h2gpu     400001    7   04
try hodlrgpu_o1ls_N400k    scal_obj1ls_hodlrgpu  400001    1   04
try h2gpu_o1ls_N800k       scal_obj1ls_h2gpu     800001    7   05
try hodlrgpu_o1ls_N800k    scal_obj1ls_hodlrgpu  800001    1   05
try h2gpu_o1ls_N1600k      scal_obj1ls_h2gpu     1600001   7   06
try hodlrgpu_o1ls_N1600k   scal_obj1ls_hodlrgpu  1600001   1   06
