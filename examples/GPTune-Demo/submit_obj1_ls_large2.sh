#!/bin/bash
# Objective 1 at the large sizes, extending the three series that submit_obj1_ls_large.sh
# left out: HODLR with finite differences, HODLR with the analytic gradient, and H2 with the
# analytic gradient.  H2 with finite differences is already measured at these sizes.
#
# Same settings as those runs, so the points extend the existing curves rather than starting new
# ones: length scale restricted to [0.02, 0.03], 64 ranks x 8 threads on 4 CPU nodes, knn 0, and
# the results written into scal_obj1ls_{h2,hodlr}cpu beside the rest of each series.
#
# One optimizer per job rather than several per job: a step that hangs or runs out of memory then
# costs only its own wall, and HODLR is expected to run out of memory somewhere in this range.
# A size that fails is dropped rather than given more nodes.  H2 goes to 25.6M, HODLR to 12.8M.
#
# GP_REPLAY_POINTS=0 switches off the held-out test-metric replay.  At 25.6M it is what killed
# both earlier runs, after their training had finished, so their statistics had to be recovered
# from the logs; it contributes nothing to the scaling figures.
#
# Rerun whenever slots free up; a row is skipped once submitted.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
LS="0.02 0.03 0.025"
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

try() { # id fmt runname optimizer NS hours
    local id=$1 fmt=$2 rn=$3 opt=$4 ns=$5 hours=$6
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=$fmt NMPI=64 NTH=8 KNN=0 "OPTIMIZER=$opt" "LENGTHSCALE=$LS" \
        GP_REPLAY_POINTS=0 RUN_NAME=$rn \
        sbatch -C cpu -N 4 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#   id                    fmt run dir              optimizer           NS        hours
try hodlrfd_o1_N3200k     1   scal_obj1ls_hodlrcpu "finite difference" 3200001   04
try hodlrgr_o1_N3200k     1   scal_obj1ls_hodlrcpu "gradient"          3200001   04
try h2gr_o1_N3200k        7   scal_obj1ls_h2cpu    "gradient"          3200001   04
try hodlrfd_o1_N6400k     1   scal_obj1ls_hodlrcpu "finite difference" 6400001   06
try hodlrgr_o1_N6400k     1   scal_obj1ls_hodlrcpu "gradient"          6400001   06
try h2gr_o1_N6400k        7   scal_obj1ls_h2cpu    "gradient"          6400001   06
try hodlrfd_o1_N12800k    1   scal_obj1ls_hodlrcpu "finite difference" 12800001  08
try hodlrgr_o1_N12800k    1   scal_obj1ls_hodlrcpu "gradient"          12800001  08
try h2gr_o1_N12800k       7   scal_obj1ls_h2cpu    "gradient"          12800001  08
# HODLR stops at 12.8M by choice: the 25.6M points are not worth their allocation.
# try hodlrfd_o1_N25600k    1   scal_obj1ls_hodlrcpu "finite difference" 25600001  12
# try hodlrgr_o1_N25600k    1   scal_obj1ls_hodlrcpu "gradient"          25600001  12
try h2gr_o1_N25600k       7   scal_obj1ls_h2cpu    "gradient"          25600001  12
