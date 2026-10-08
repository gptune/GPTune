#!/bin/bash
# CPU runs of the 1D Schwefel comparison (objective 6), to put beside the GPU ones: H2 and HODLR,
# the same 4 ranks and the same tolerances, on one CPU node (16 threads per rank).  Rerun whenever
# slots free up; a row is skipped once submitted (marker h2_unstructured_cmp/.sub_<id>).
#
# It never uses the whole queue: MAXJOBS (default 4 of the 5 the QOS allows) leaves a slot for
# other sessions, counting this user's jobs in the same QOS.
cd "$(dirname "$0")"
MAXJOBS=${MAXJOBS:-4}
NV="1e-3 1e-1 1e-2"; LS="1e-2 2.718281828 0.1"
echo "== $(date)"
# when run from scrontab, sbatch would pass that job's SLURM_* variables to the new job
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

mine() { squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx "$1"; }

try() { # id runname NS fmt ranks threads [optimizers] [hours]
    local id=$1 rn=$2 ns=$3 fmt=$4 ranks=${5:-4} threads=${6:-16}
    local opt=${7:-"gradient;finite difference"} hours=${8:-04}
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    local used=$(mine premium)
    if [ "$used" -ge "$MAXJOBS" ]; then echo "$id: holding ($used of $MAXJOBS premium slots used)"; return; fi
    out=$(env USE_GPU=0 FORMAT=$fmt NMPI=$ranks NTH=$threads "OPTIMIZER=$opt" \
        "NOISEVAR=$NV" "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -C cpu -N 1 -q premium -t ${hours}:00:00 test_GP_h2_unstructured.sh iso $ns 6 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}

# HODLR on CPU runs 64 ranks x 2 threads (the whole node in MPI), H2 on CPU 4 ranks x 16 threads
# as the GPU runs do.
#   id                run dir             NS        format ranks threads
try h2cpu_1d_N100k    scal_1d_h2cpu       100001   7       4     16
try hodlrcpu_1d_N100k scal_1d_hodlrcpu    100001   1      64      2
try h2cpu_1d_N200k    scal_1d_h2cpu       200001   7       4     16
try hodlrcpu_1d_N200k scal_1d_hodlrcpu    200001   1      64      2
try h2cpu_1d_N400k    scal_1d_h2cpu       400001   7       4     16
try hodlrcpu_1d_N400k scal_1d_hodlrcpu    400001   1      64      2
try h2cpu_1d_N800k    scal_1d_h2cpu       800001   7       4     16
try hodlrcpu_1d_N800k scal_1d_hodlrcpu    800001   1      64      2

# HODLR on CPU spends a whole 4 h wall inside the gradient step before finite differences can run
# (229 s per gradient evaluation at 100k), so the larger sizes take the finite-difference step alone.
try hodlrcpu_1d_N200k_fd scal_1d_hodlrcpu 200001   1      64      2 "finite difference" 04
try hodlrcpu_1d_N400k_fd scal_1d_hodlrcpu 400001   1      64      2 "finite difference" 06
try hodlrcpu_1d_N800k_fd scal_1d_hodlrcpu 800001   1      64      2 "finite difference" 08
