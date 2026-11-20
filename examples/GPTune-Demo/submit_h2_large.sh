#!/bin/bash
# Larger H2 points for the paper's 2D and 3D GPU scaling figures (the H2 rows of tab:h2gpu end at 800k in 2D,
# where the table stops at 400k, and at 200k in 3D): 2D at N=1.6M (isotropic objective 2, anisotropic objective
# 4) and 3D anisotropic (objective 5) at N=400k. Gradient and finite-difference L-BFGS and MCMC, no MALA. The
# MCMC sampling-time cap is 3x the shorter L-BFGS training time, as in the table's runs.
#
# Build: origin/H2_optimization 1d3d6d1 (19 Nov 2026 23:24; staged as its own worktree, ButterflyPACK_1d3d6d1, with
# the cmake options of the eb5e78b build): on top of eb5e78b's GPU heap (operators evicted, finished parent blocks
# moved to the host) the reclaimer spills the kept F22 records, also those of the other live operators, and the
# sketch arena and the partial-factorization CB arenas go in chunks that fit the heap. A first submission on
# eb5e78b (jobs 61633090/2/3) was cancelled before it started. The table's 2D iso and raw-coordinate rows ran 3f05cd5
# (1 Oct 2026); its earlier 3D 400k attempts (jobs 59314687/8 at 8 ranks, 59318027/59319833 at 64 ranks, 40 GB
# A100s) exhausted the GPU heap. The anisotropic rows use the scaled-coordinate tree (case aniso_scaled), as the
# table's anisotropic runs do; both go on 80 GB A100s: the scaled 2D run at 400k died of a 32 GB transient
# of the adaptive ID (frame blocks on a strip geometry) and the scaled 3D run at 200k of a 17 GB raw fill, both
# on 40 GB GPUs. Layouts: 2D 16 ranks on 4 nodes, as at 800k (power of 4); 3D 8 ranks on 2 nodes (power of 8),
# as the H2 400k run of the H2-vs-INLA table (tab:3d), with nmin_leaf 32 and H2_REGION_IMBALANCE 8 as in the
# scaled 3D 100k run (see submit_aniso_scaled.sh).
#
# Submitted to gpu_regular on m4293_g (jobs 61633367/8/9); later runs charge m5449_g premium (the premium QOS allows
# 5 submitted jobs, shared with other sessions).
# Idempotent: a row is skipped once submitted (marker h2_unstructured_cmp/.sub_<id>).
cd "$(dirname "$0")"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
NV="1e-3 1e-1 1e-2"; LS="1e-2 2.718281828 0.1"; OPT='gradient;finite difference;mcmc'
export BPACK_DIR=${BPACK_DIR:-$PWD/../ButterflyPACK/ButterflyPACK_1d3d6d1}
try() { # id runname case obj NS nmpi rpn nodes hours constraint nmin_leaf region_imbalance
    local id=$1 rn=$2 case=$3 obj=$4 ns=$5 nmpi=$6 rpn=$7 nodes=$8 hours=$9 cons=${10} nmin=${11:-64} imb=${12:-1.15}
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    out=$(env USE_GPU=1 FORMAT=7 NMPI=$nmpi RANKS_PER_NODE=$rpn NMIN_LEAF=$nmin H2_REGION_IMBALANCE=$imb "OPTIMIZER=$OPT" "NOISEVAR=$NV" "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -A m5449_g -C "$cons" --gpus-per-node=4 -N $nodes -q premium -t $hours:00:00 test_GP_h2_unstructured.sh $case $ns $obj 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}
#   id                  run dir                      case          obj NS       nmpi rpn nodes h  constraint   nmin imb
try h2_2di_N1600k     scal_h2gpu_2d_iso            iso           2   1600001  16   4   4     6  gpu
try h2_2da_s_N1600k   scal_h2gpu_2d_aniso_scaled   aniso_scaled  4   1600001  16   4   4     8  gpu\&hbm80g
try h2_3da_s_N400k    scal_h2gpu_3d_aniso_scaled   aniso_scaled  5   400001   8    4   2     12 gpu\&hbm80g  32   8
