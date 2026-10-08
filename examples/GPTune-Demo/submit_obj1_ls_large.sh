#!/bin/bash
# Objective 1 at the large sizes, 3.2M to 25.6M, H2 only and finite difference only, on both CPU
# and GPU.  HODLR is left out: it is 13x the cost of H2 per likelihood evaluation at 1.6M on CPU,
# so these sizes are out of reach for it.
#
# The length scale is restricted to [0.02, 0.03] as at 1.6M.  Below 1.6M the CPU H2 series is
# unrestricted, but finite difference never moves off l = 1 when it is unrestricted, so these
# points would measure the wrong regime without the restriction.
#
# The layouts are the ones the rest of each series uses, so these extend the curves rather than
# starting new ones: CPU 64 ranks x 8 threads on 4 nodes, GPU 8 ranks on 8 A100s over 2 nodes.
# Resources stay fixed as N grows, which is the point of a scaling curve; a size that runs out of
# memory is dropped rather than given more nodes.
#
# GP_REPLAY_POINTS=2 cuts the post-run test-metric replay from 20 refactorisations to 2.  Those
# metrics do not enter the scaling figures and at these sizes they would cost more than the
# training they describe.
#
# Rerun whenever slots free up; a row is skipped once submitted.  CPU and GPU draw on separate QOS
# pools (premium and gpu_premium), each capped to leave a slot for other sessions.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
LS="0.02 0.03 0.025"
echo "== $(date)"
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

used() { squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx "$1"; }

try_cpu() { # id NS hours
    local id=$1 ns=$2 hours=$3
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local u=$(used premium)
    if [ "$u" -ge "$MAXJOBS" ]; then echo "$id: holding ($u of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=7 NMPI=64 NTH=8 KNN=0 "OPTIMIZER=finite difference" "LENGTHSCALE=$LS" \
        GP_REPLAY_POINTS=2 RUN_NAME=scal_obj1ls_h2cpu \
        sbatch -C cpu -N 4 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

try_gpu() { # id NS hours
    local id=$1 ns=$2 hours=$3
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local u=$(used gpu_premium)
    if [ "$u" -ge "$MAXJOBS" ]; then echo "$id: holding ($u of $MAXJOBS gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=7 NMPI=8 RANKS_PER_NODE=4 KNN=0 "OPTIMIZER=finite difference" "LENGTHSCALE=$LS" \
        GP_REPLAY_POINTS=2 RUN_NAME=scal_obj1ls_h2gpu \
        sbatch -C gpu --gpus-per-node=4 -N 2 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 1 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

#        id                     NS         hours
try_cpu  h2cpu_o1ls_N3200k      3200001    03
try_gpu  h2gpu_o1ls_N3200k      3200001    03
try_cpu  h2cpu_o1ls_N6400k      6400001    04
try_gpu  h2gpu_o1ls_N6400k      6400001    04
try_cpu  h2cpu_o1ls_N12800k     12800001   06
try_gpu  h2gpu_o1ls_N12800k     12800001   06
try_cpu  h2cpu_o1ls_N25600k     25600001   10
try_gpu  h2gpu_o1ls_N25600k     25600001   10
