#!/bin/bash
# 3D at 100k and 200k under exactly the settings the 400k runs used, so that the three sizes form
# one scaling curve.  The existing 100k and 200k points cannot be compared with the 400k ones:
# they predate H2_compress_factors and their optimizers settled at much shorter length scales
# (aniso at 0.13 against 1.8-2.7), where the ranks are far larger.  Taken together they show the
# cost falling as N doubles, which is an artefact of those two differences.
#
# Matched here: H2_COMPRESS=1, LENGTHSCALE floored at 0.3, noise left at GPTune's default,
# 8 ranks 2 per node over 4 nodes, 40 GB cards, finite differences.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
# What broke the earlier aniso runs was the INITIAL length scale, not the floor.  Starting the
# three dimensions at 1.0 puts the model where the likelihood is +4.5e9; finite differences then
# makes one jump onto whatever the floor is and stops after 4 iterations, whatever that floor is
# (0.3 and 0.05 both failed the same way).  Starting at 0.1 begins at -1.3e5 and walks to
# (0.129, 2.718, 0.432) over 17 iterations -- the true answer, since the terms
# sin(10 pi x1) + 4 x2^2 + sin(3 pi x3) want about 0.13, unbounded and 0.43.
#
# So both objectives go back to GPTune's original range for this campaign, 1e-2 to e starting at
# 0.1, which is low enough to stay clear of the l -> 0 collapse that crashed the old code and is
# what the runs that converged used.  Rank follows the volume l1 l2 l3 here, and N l1 l2 l3 is
# 6e4 at the true optimum, well past the peak at 1e3, so no floor above 1e-2 is needed.
LS_ISO="1e-2 2.718281828 0.1"
LS_ANISO="1e-2 2.718281828 0.1"
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id runname case objtype NS hours
    local id=$1 rn=$2 case=$3 obj=$4 ns=$5 hours=$6
    local LS=$([ "$case" = aniso ] && echo "$LS_ANISO" || echo "$LS_ISO")
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=7 NMPI=8 RANKS_PER_NODE=2 H2_COMPRESS=1 "OPTIMIZER=gradient;finite difference" \
        "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -A m4293_g -C "gpu&hbm40g" --gpus-per-node=4 -N 4 -q premium -t ${hours}:00:00 \
        test_GP_h2_unstructured.sh $case $ns $obj 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                  run dir                 case  obj NS      hours
try h2_3da_100k_g       comp_3d_aniso_100k_g    aniso 5   100001  04
try h2_3da_200k_g       comp_3d_aniso_200k_g    aniso 5   200001  05
try h2_3da_400k_g       comp_3d_aniso_400k_g    aniso 5   400001  06
try h2_3di_100k_g       comp_3d_iso_100k_g      iso   3   100001  04
try h2_3di_200k_g       comp_3d_iso_200k_g      iso   3   200001  05
try h2_3di_400k_g       comp_3d_iso_400k_g      iso   3   400001  06
