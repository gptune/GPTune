#!/bin/bash
# The anisotropic rows of the paper's H2/HODLR GPU table (tab:h2gpu) again, with the cluster tree built on
# the coordinates divided by the length scales (case aniso_scaled; H2 then needs --h2_unstructured 1, which
# the case sets). The first runs (scal_*gpu_*d_aniso) used raw coordinates for both formats. Same
# objectives, sizes, layouts, noise and length-scale ranges and training algorithms as those runs; the
# current build also gives every H2 ID the absolute floor H2_s_op (HODLR ignores it, and it would not
# change HODLR's ranks: every HODLR off-diagonal block holds the interface of two adjacent clusters).
#
# Idempotent: a row is skipped once submitted (marker h2_unstructured_cmp/.sub_<id>), and held while four
# gpu_premium jobs of this user are queued or running (the QOS allows five; one stays free for other
# sessions). Rerun until every row is submitted: bash submit_aniso_scaled.sh   (or with --loop)
cd "$(dirname "$0")"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
NV="1e-3 1e-1 1e-2"; LS="1e-2 2.718281828 0.1"; ALL='gradient;finite difference;mcmc;mala'
# The build: origin/H2_optimization eb5e78b staged as its own worktree (its GPU heap evicts operators and
# moves finished parent blocks to the host where an allocation does not fit). Rows submitted before
# 19 Nov 2026 16:30 (H2 2D 100k-400k, HODLR 2D 100k/200k, HODLR 3D 25k) ran eef6ed0 in the main checkout.
export BPACK_DIR=${BPACK_DIR:-$PWD/../ButterflyPACK/ButterflyPACK_eb5e78b}
pending=0
try() { # id runname obj NS nmpi rpn nodes hours format optimizer nmin_leaf region_imbalance
    local id=$1 rn=$2 obj=$3 ns=$4 nmpi=$5 rpn=$6 nodes=$7 hours=$8 fmt=$9 opt=${10} nmin=${11:-64} imb=${12:-1.15}
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    [ -n "$ONLY" ] && [ "${id#$ONLY}" = "$id" ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "${MAXJOBS:-4}" ]; then echo "$id: holding ($used of ${MAXJOBS:-4} gpu_premium slots used)"; pending=$((pending + 1)); return; fi
    out=$(env USE_GPU=1 FORMAT=$fmt NMPI=$nmpi RANKS_PER_NODE=$rpn NMIN_LEAF=$nmin H2_REGION_IMBALANCE=$imb "OPTIMIZER=$opt" "NOISEVAR=$NV" "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -A m5449_g -C gpu --gpus-per-node=4 -N $nodes -q premium -t $hours:00:00 test_GP_h2_unstructured.sh aniso_scaled $ns $obj 2>&1 | tail -1)
    echo "$id: $out"
    if echo "$out" | grep -q "Submitted"; then echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id; else pending=$((pending + 1)); fi
}
submit_all() {
    pending=0
    echo "== $(date)"
    # (H2 in 3D: nmin_leaf 32, since 64 lets an unstructured leaf of scaled 3D points pass the GPU's
    #  512-point near-block cap, e.g. 518 points at N=100k, l=(0.12,0.9,0.3); 2D stays at 64)
    # (H2 in 3D: H2_REGION_IMBALANCE 8, so the 8 ranks keep the equal cubes of the raw runs (active ranks
    #  1 1 8 8) wherever no cube is empty. With the default 1.15 a slightly anisotropic l, e.g. inverse
    #  length scales (10.7, 9.8, 9.8), makes the cubes 1.2x uneven, the planner takes Morton ranges with the
    #  first reduction a level finer (1 1 1 8), and levels 2 and 1 then run on rank 0 alone: job 61611963
    #  exhausted its 33 GB GPU heap at level 1 (8.7 GB kept factors + a 17.8 GB parent-block request).)
    #   id                    run dir                          obj NS      nmpi rpn nodes h fmt optimizer nmin imb
    try hodlr_2da_s_N100k  scal_hodlrgpu_2d_aniso_scaled   4   100001  8    4   2     6 1   "$ALL"
    try h2_2da_s_N100k     scal_h2gpu_2d_aniso_scaled      4   100001  4    4   1     4 7   "$ALL"
    try hodlr_2da_s_N200k  scal_hodlrgpu_2d_aniso_scaled   4   200001  8    4   2     6 1   "$ALL"
    try h2_2da_s_N200k     scal_h2gpu_2d_aniso_scaled      4   200001  4    4   1     4 7   "$ALL"
    try h2_3da_s_N100k     scal_h2gpu_3d_aniso_scaled      5   100001  8    4   2     6 7   "$ALL"  32   8
    try hodlr_3da_s_N25k   scal_hodlrgpu_3d_aniso_scaled   5   25001   8    4   2     4 1   "$ALL"
    try h2_2da_s_N400k     scal_h2gpu_2d_aniso_scaled      4   400001  4    4   1     4 7   "$ALL"
    try hodlr_2da_s_N400k  scal_hodlrgpu_2d_aniso_scaled   4   400001  16   4   4     6 1   "$ALL"
    # (200k: nmin_leaf 50, not 32. The base depth comes from N alone, round(N^(1/3)) against
    #  round(nmin^(1/3)): 32 gives 5 levels at 200k (4 at 100k), and job 61619701 exhausted the heap in its
    #  first factorization (17 GB of raw fill at a transition); 50 keeps the raw run's 4 levels at the
    #  isotropic start (50 * 4096 > 200001) and, in a simulation over a 12^3 grid of l, overflows the 512-point
    #  leaf cap at the same corners as 32 (315 of 1728, against 396 for 64).)
    try h2_3da_s_N200k     scal_h2gpu_3d_aniso_scaled      5   200001  8    2   4     6 7   "$ALL"  50   8
    # (50k, added 20 Nov 2026 for the 3D GPU scaling figure: the 100k layout and settings. nmin_leaf 32 gives
    #  the same 4 base levels as at 100k (round(50001^(1/3)) = 37 against 3*2^4 = 48), so the leaves hold
    #  about half the points of the 100k ones. Submitted on 1d3d6d1, as the 400k run of submit_h2_large.sh:
    #  BPACK_DIR=$PWD/../ButterflyPACK/ButterflyPACK_1d3d6d1 ONLY=h2_3da_s_N50k bash submit_aniso_scaled.sh --loop)
    try h2_3da_s_N50k      scal_h2gpu_3d_aniso_scaled      5   50001   8    4   2     4 7   "$ALL"  32   8
}
submit_all
if [ "$1" = "--loop" ]; then
    # retry every 10 minutes until every row is submitted, for at most two days
    for _ in $(seq 288); do
        [ "$pending" -eq 0 ] && break
        sleep 600
        submit_all
    done
fi
echo "rows still pending: $pending"
