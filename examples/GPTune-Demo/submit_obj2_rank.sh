#!/bin/bash
# Objective 2, the 2D isotropic function, with an isotropic kernel, so the model has one length
# scale as in 1D and the rank can be plotted against a single N l^d.  HODLR and H2 on CPU.
#
# The point of these runs is the rank, not the timing, so they differ from the scaling campaigns:
#
#   the length scale is left unrestricted.  Its default range spans l from 1e-5 to 2.7, and the
#   optimizers wander over it, which is what gives the rank plot its x axis; a restricted range
#   would collapse to a single column.
#   VERBOSITY=1 makes the workers log rankmax_of_level for every build.  It is the only way to see
#   the H2 construction ranks, whose init_compute counter stays at zero whatever the verbosity.
#   Three sizes, so that the collapse onto N l^d can be checked rather than assumed: if it is the
#   right variable, runs of different N at the same N l^d report the same rank.
#
# 64 ranks over 4 nodes, as the 1D CPU runs used; 2D wants a rank count that is a power of 4 and
# 64 is 4^3.  Rerun whenever slots free up; a row is skipped once submitted.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
OPT="gradient;finite difference;mcmc"
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id runname NS fmt hours
    local id=$1 rn=$2 ns=$3 fmt=$4 hours=$5
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=$fmt NMPI=64 NTH=8 KNN=0 VERBOSITY=1 "OPTIMIZER=$OPT" RUN_NAME=$rn \
        sbatch -C cpu -N 4 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 2 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                  run dir             NS       fmt hours
try h2_o2rank_N100k     rank_obj2_h2cpu     100001    7   04
try hodlr_o2rank_N100k  rank_obj2_hodlrcpu  100001    1   04
try h2_o2rank_N200k     rank_obj2_h2cpu     200001    7   05
try hodlr_o2rank_N200k  rank_obj2_hodlrcpu  200001    1   05
try h2_o2rank_N400k     rank_obj2_h2cpu     400001    7   06
try hodlr_o2rank_N400k  rank_obj2_hodlrcpu  400001    1   06
