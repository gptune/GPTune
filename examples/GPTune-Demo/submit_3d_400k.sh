#!/bin/bash
# Does H2 reach N=400k in 3D now?  Both 3D objectives, isotropic (objective 3) and anisotropic
# (objective 5), with the H2 code pulled on 13 October.
#
# Every earlier attempt at this size died with "GPU heap exhausted ... use more ranks", at 8 ranks
# and again at 64, because the levels near the tree root stay on one rank and more ranks do not
# shrink its peak.  What is new is H2_compress_factors: nvCOMP compression of the factors kept on
# the device, which is what lets more boxes share a GPU.  It makes the factorization lossy within
# tol_comp, so these runs are not bitwise comparable with the earlier ones.
#
# Four rows: each objective on 40 GB and on 80 GB cards.  40 GB is the real question, since 80 GB
# was already the expected escape route.  8 ranks, 2 per node over 4 nodes, the layout the earlier
# 3D runs used above N=100k.  Finite differences alone: it is the cheapest evaluation and the one
# that needs least memory, so if it cannot fit, nothing else will.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id runname case objtype card hours
    local id=$1 rn=$2 case=$3 obj=$4 card=$5 hours=$6
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=7 NMPI=8 RANKS_PER_NODE=2 H2_COMPRESS=1 "OPTIMIZER=finite difference" \
        RUN_NAME=$rn \
        sbatch -A m5449_g -C "gpu&${card}" --gpus-per-node=4 -N 4 -q premium -t ${hours}:00:00 \
        test_GP_h2_unstructured.sh $case 400001 $obj 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                  run dir                    case  obj card    hours
try h2_3di_400k_40g     comp_3d_iso_400k_40g       iso   3   hbm40g  04
try h2_3da_400k_40g     comp_3d_aniso_400k_40g     aniso 5   hbm40g  04
try h2_3di_400k_80g     comp_3d_iso_400k_80g       iso   3   hbm80g  04
try h2_3da_400k_80g     comp_3d_aniso_400k_80g     aniso 5   hbm80g  04

# The rows above take GPTune's default hyperparameter ranges, which let the length scale reach
# 1.01e-5.  Both corners of that range are unusable at this size:
#
#   l -> 0     the isotropic run walked down to 1.3e-5 and the workers died in
#              collect_lazy_far_sources.  That was a bookkeeping bug (a full-rank box did not send
#              its wave), fixed upstream in ea023e6 and verified gone here.
#   l ~ 0.1    the rank of the off-diagonal blocks peaks near N l^3 = 1e3, which at N = 400k is
#              l = 0.136.  The standalone reproducer exhausts a 40 GB card there at tol 1e-10, and
#              64 ranks do not help: reduction_threshold 8 drops the active processes from 64 to 8
#              at level 2, so the near-root work does not distribute.
#
# These rows therefore floor the length scale at 0.3, where N l^3 = 1.1e4 is an order of magnitude
# past the peak and the ranks are small.  The noise range is left at GPTune's default, as in the
# aniso run that completed; only the length scale is restricted.
LS="0.3 2.718281828 1.0"
tryr() { # id runname case objtype card hours -- as try, with the campaign's ranges
    local id=$1 rn=$2 case=$3 obj=$4 card=$5 hours=$6
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=7 NMPI=8 RANKS_PER_NODE=2 H2_COMPRESS=1 "OPTIMIZER=finite difference" \
        "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -A m5449_g -C "gpu&${card}" --gpus-per-node=4 -N 4 -q premium -t ${hours}:00:00 \
        test_GP_h2_unstructured.sh $case 400001 $obj 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}
tryr h2_3di_400k_40g_ls  comp_3d_iso_400k_40g_ls    iso   3   hbm40g  04
tryr h2_3da_400k_40g_ls  comp_3d_aniso_400k_40g_ls  aniso 5   hbm40g  04
