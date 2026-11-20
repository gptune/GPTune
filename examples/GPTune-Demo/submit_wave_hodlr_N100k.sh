#!/bin/bash
# HODLR GPU wave 1 (N=100k) submissions that hit the gpu_premium submit limit on 2026-10-02.
# Idempotent: a case is skipped when its run directory already has a worker log or a job of its
# name is queued. Run whenever slots free up: bash submit_wave_hodlr_N100k.sh
cd "$(dirname "$0")"
OPTS='gradient;finite difference;mcmc;mala'; NV="1e-3 1e-1 1e-2"; LS="1e-2 2.718281828 0.1"
try() { # name case objtype nmpi nodes
    local name=$1 case=$2 obj=$3 nmpi=$4 nodes=$5
    [ -e h2_unstructured_cmp/.submitted_$name ] && { echo "$name: already submitted (job $(cat h2_unstructured_cmp/.submitted_$name))"; return; }
    env USE_GPU=1 FORMAT=1 NMPI=$nmpi RANKS_PER_NODE=4 "OPTIMIZER=$OPTS" "NOISEVAR=$NV" "LENGTHSCALE=$LS" RUN_NAME=$name \
        sbatch -A m5449_g -C gpu --gpus-per-node=4 -N $nodes -q premium -t 04:00:00 --job-name=GPTune_H2_unstr \
        test_GP_h2_unstructured.sh $case 100001 $obj > /tmp/sub_$name.out 2>&1
    cat /tmp/sub_$name.out | sed "s/^/$name: /" | tail -1
    grep -q "Submitted batch job" /tmp/sub_$name.out && grep -o "[0-9]*$" /tmp/sub_$name.out > h2_unstructured_cmp/.submitted_$name
}
try scal_hodlrgpu_2d_iso   iso   2 4 1
try scal_hodlrgpu_2d_aniso aniso 4 4 1
try scal_hodlrgpu_3d_iso   iso   3 8 2
try scal_hodlrgpu_3d_aniso aniso 5 8 2
