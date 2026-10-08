#!/bin/bash
# Pending submissions of the H2/HODLR GPU scaling campaign, tried in priority order; rerun any time.
# A row is skipped once submitted (marker h2_unstructured_cmp/.sub_<id>). gpu_premium submit limits
# reject the rest; just run again later.
cd "$(dirname "$0")"
echo "== $(date)"
# When this script runs from scrontab it is itself inside a small Slurm job, and sbatch would
# propagate that job's SLURM_* variables into the new job, where they cap the worker step
# ("srun: error: ... More processors requested than permitted").  Drop them all.
for v in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
NV="1e-3 1e-1 1e-2"; LS="1e-2 2.718281828 0.1"; ALL='gradient;finite difference;mcmc;mala'
try() { # id runname case obj NS nmpi rpn nodes hours format optimizer
    local id=$1 rn=$2 case=$3 obj=$4 ns=$5 nmpi=$6 rpn=$7 nodes=$8 hours=$9 fmt=${10} opt=${11}
    [ -e h2_unstructured_cmp/.sub_$id ] && return
    # leave a slot of the 5 the QOS allows for other sessions
    local used=$(squeue -u $USER -h -o "%q" 2>/dev/null | grep -cx gpu_premium)
    if [ "$used" -ge "${MAXJOBS:-4}" ]; then echo "$id: holding ($used of ${MAXJOBS:-4} gpu_premium slots used)"; return; fi
    out=$(env USE_GPU=1 FORMAT=$fmt NMPI=$nmpi RANKS_PER_NODE=$rpn "OPTIMIZER=$opt" "NOISEVAR=$NV" "LENGTHSCALE=$LS" RUN_NAME=$rn \
        sbatch -C gpu --gpus-per-node=4 -N $nodes -q premium -t $hours:00:00 test_GP_h2_unstructured.sh $case $ns $obj 2>&1 | tail -1)
    echo "$id: $out"
    echo "$out" | grep -q "Submitted" && echo "$out" | grep -o "[0-9]*$" > h2_unstructured_cmp/.sub_$id
}
#   id                      run dir                  case  obj NS     nmpi rpn nodes h fmt optimizer
# Retries after "H2 GPU heap exhausted" (the library's advice: use more ranks).  3D needs a
# power-of-8 rank count, so 8 -> 64 ranks; 2D a power of 4, so 16 -> 64.  HODLR 2D at N=400k is
# dropped per the user's policy (HODLR is expected to run out of memory at large N).
try h2_3di_N200k_mala64    scal_h2gpu_3d_iso        iso   3  200001 64 4 16   4 7  "mala"
try h2_3da_N200k_mala64    scal_h2gpu_3d_aniso      aniso 5  200001 64 4 16   4 7  "mala"
try hodlr_3da_N25k_r64     scal_hodlrgpu_3d_aniso   aniso 5  25001  64 4 16   2 1  "gradient"
try h2_3di_N400k_r64       scal_h2gpu_3d_iso        iso   3  400001 64 4 16   6 7  "$ALL"
try h2_3da_N400k_r64       scal_h2gpu_3d_aniso      aniso 5  400001 64 4 16   6 7  "$ALL"
try h2_2di_N800k           scal_h2gpu_2d_iso        iso   2  800001 16 4 4    6 7  "$ALL"
try h2_2da_N800k           scal_h2gpu_2d_aniso      aniso 4  800001 16 4 4    6 7  "$ALL"

# 1D Schwefel (objective 6): HODLR against H2 at one layout for every size, 4 ranks over 2 nodes
# (2 per node, one A100 each, so each rank has half a node's host memory), three optimizers.
try h2_1d_N100k     scal_1d_h2gpu       iso   6  100001   4  2 2    2 7  "gradient;finite difference;mcmc"
try hodlr_1d_N100k  scal_1d_hodlrgpu   iso   6  100001   4  2 2    2 1  "gradient;finite difference;mcmc"
try h2_1d_N200k     scal_1d_h2gpu       iso   6  200001   4  2 2    2 7  "gradient;finite difference;mcmc"
try hodlr_1d_N200k  scal_1d_hodlrgpu   iso   6  200001   4  2 2    2 1  "gradient;finite difference;mcmc"
try h2_1d_N400k     scal_1d_h2gpu       iso   6  400001   4  2 2    3 7  "gradient;finite difference;mcmc"
try hodlr_1d_N400k  scal_1d_hodlrgpu   iso   6  400001   4  2 2    3 1  "gradient;finite difference;mcmc"
try h2_1d_N800k     scal_1d_h2gpu       iso   6  800001   4  2 2    3 7  "gradient;finite difference;mcmc;mala"
try hodlr_1d_N800k  scal_1d_hodlrgpu   iso   6  800001   4  2 2    3 1  "gradient;finite difference;mcmc;mala"
try h2_1d_N1600k    scal_1d_h2gpu       iso   6  1600001  4  2 2    4 7  "gradient;finite difference;mcmc;mala"
try hodlr_1d_N1600k scal_1d_hodlrgpu   iso   6  1600001  4  2 2    4 1  "gradient;finite difference;mcmc;mala"

# MALA for the 1D runs that were submitted before MALA was added to the set
try h2_1d_N100k_mala scal_1d_h2gpu      iso   6  100001   4  2 2    2 7  "mala"
try hodlr_1d_N100k_mala scal_1d_hodlrgpu   iso   6  100001   4  2 2    2 1  "mala"
try h2_1d_N200k_mala scal_1d_h2gpu      iso   6  200001   4  2 2    2 7  "mala"
try hodlr_1d_N200k_mala scal_1d_hodlrgpu   iso   6  200001   4  2 2    2 1  "mala"
try h2_1d_N400k_mala scal_1d_h2gpu      iso   6  400001   4  2 2    3 7  "mala"
try hodlr_1d_N400k_mala scal_1d_hodlrgpu   iso   6  400001   4  2 2    3 1  "mala"

# the CPU halves of the 1D comparisons, each with its own cap on the premium queue
bash "$(dirname "$0")/submit_1d_cpu.sh"
bash "$(dirname "$0")/submit_obj1_cpu.sh"
bash "$(dirname "$0")/submit_obj1_ls_cpu.sh"
bash "$(dirname "$0")/submit_obj1_ls_gpu.sh"
bash "$(dirname "$0")/submit_obj1_ls_large.sh"
