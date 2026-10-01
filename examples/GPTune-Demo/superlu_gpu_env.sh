# Sourced by the batch scripts (run_superlu_gpu.sbatch, run_scaling_gpu.sbatch) that run
# model_comparison_updated_superlu.py with SuperLU_DIST workers on 4 GPU nodes: 16 MPI ranks on a 4x4
# process grid, 4 ranks per node, one A100 per rank. The driver runs on the first node and talks to the
# workers through the CONTROL/DATA/RESULT files. Needs GPTUNE_ROOT, DEMO_DIR and OBJ_CHOICE.

cd $GPTUNE_ROOT
. ./run_env.sh
module load cudatoolkit/13.2
module load craype-accel-nvidia80


#MPI settings:
#################################################
NROW=4   # number of MPI row processes
NCOL=4   # number of MPI column processes
NPZ=1    # number of 2D process grids
NTH=1 # number of OMP threads
NNODE=4
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
export OMP_NUM_THREADS=$NTH
DRIVER_OMP_THREADS=${DRIVER_OMP_THREADS:-32} # OpenMP threads of the driver (python) process on the first node
#################################################


#SUPERLU settings:
#################################################
# CUDA build of SuperLU_DIST, see superlu_dist_gpu/build/configure_gpu.sh
export SUPERLU_PYTHON_LIB_PATH=$GPTUNE_ROOT/examples/SuperLU_DIST/superlu_dist_gpu/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=1 # whether to do CPU or GPU numerical factorization
export GPU3DVERSION=0 # whether to do use the latest C++ numerical factorization
export SUPERLU_ACC_SOLVE=0 # whether to do CPU or GPU triangular solve
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU
export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=100000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
export SUPERLU_MPI_PROCESS_PER_GPU=1
export SUPERLU_REUSE_PATTERN=${SUPERLU_REUSE_PATTERN:-1} # refactor reusing the ordering and symbolic factorization when the sparsity pattern (cutoff) does not change
# SuperLU communicates host buffers only; the Python executables are not linked with the Cray GTL library
export MPICH_GPU_SUPPORT_ENABLED=0
#################################################


#ButterflyPACK settings:
#################################################
export BPACK_PYTHON_LIB_PATH=$GPTUNE_ROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
#################################################

echo "JOB=$SLURM_JOB_ID NODES=$SLURM_JOB_NODELIST OBJ_CHOICE=$OBJ_CHOICE"
echo "PYTHON=$(which python) SUPERLU_PYTHON_LIB_PATH=$SUPERLU_PYTHON_LIB_PATH"
echo "SUPERLU_ACC_OFFLOAD=$SUPERLU_ACC_OFFLOAD SUPERLU_ACC_SOLVE=$SUPERLU_ACC_SOLVE SUPERLU_REUSE_PATTERN=$SUPERLU_REUSE_PATTERN DRIVER_OMP_THREADS=$DRIVER_OMP_THREADS"


# superlu_session LOG_PREFIX: start the SuperLU workers, run the driver in the current directory
# (logs LOG_PREFIX_driver.log and LOG_PREFIX_worker.log), shut the workers down; returns the driver's exit status
superlu_session () {
    local log_prefix=$1
    # files used by the superlu file interface, kept on scratch
    local run_dir=$PSCRATCH/gptune_superlu_gpu/$SLURM_JOB_ID/$log_prefix
    mkdir -p $run_dir
    export CONTROL_FILE=$run_dir/control.txt
    export DATA_FILE=$run_dir/data.bin
    export RESULT_FILE=$run_dir/result.bin
    rm -f $CONTROL_FILE $DATA_FILE $RESULT_FILE
    echo "$(date +%T) $log_prefix: GP_NS=${GP_NS:-} GP_OPTIMIZERS=${GP_OPTIMIZERS:-} in $(pwd)"

    srun -N $NNODE -n $NCORE_VAL_TOT --ntasks-per-node=4 -c 32 --cpu-bind=cores --gpus-per-task=1 \
        python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c $NCOL -r $NROW -d $NPZ -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 0 -n 0 \
        > ${log_prefix}_worker.log 2>&1 &
    local worker_pid=$!

    # the driver assembles K and dK/dtheta with OpenMP while the workers are idle
    printf "${OBJ_CHOICE}\n" | OMP_NUM_THREADS=$DRIVER_OMP_THREADS python -u ${DRIVER_SCRIPT:-$DEMO_DIR/model_comparison_updated_superlu.py} > ${log_prefix}_driver.log 2>&1 &
    local driver_pid=$!

    # the driver waits forever on the control file if the workers die, so stop it in that case
    while kill -0 $driver_pid 2>/dev/null; do
        if ! kill -0 $worker_pid 2>/dev/null; then
            echo "SuperLU workers exited before the driver finished, stopping the driver"
            kill $driver_pid
            break
        fi
        sleep 10
    done
    wait $driver_pid
    local driver_status=$?
    echo "$(date +%T) $log_prefix: driver exit status: $driver_status"

    # a worker that is in the middle of an operation overwrites the command with "done" when it finishes,
    # so repeat the terminate command until the workers exit, and kill them after 10 minutes
    for ((i = 0; i < 120; ++i)); do
        kill -0 $worker_pid 2>/dev/null || break
        python -c "from pdbridge import *; superlu_terminate()"
        sleep 5
    done
    kill $worker_pid 2>/dev/null
    wait $worker_pid
    echo "$(date +%T) $log_prefix: worker exit status: $?"
    return $driver_status
}
