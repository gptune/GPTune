#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --time=00:25:00
#SBATCH --job-name=slu-scaling
#SBATCH --output=bench_superlu_scaling_%j.out

# CPU-only SuperLU_DIST rank/thread scaling for the N=100k sparse-GP matrix (bench_superlu_scaling.py),
# using the same driver <-> worker file interface as model_comparison_updated_superlu.py.
# Each configuration starts its own workers, replays 3 factor+logdet+solve rounds and shuts them down.

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo

cd $GPTUNE_ROOT
. ./run_env.sh
cd $DEMO_DIR

# CPU build of SuperLU_DIST
export SUPERLU_PYTHON_LIB_PATH=$GPTUNE_ROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=0
export SUPERLU_ACC_SOLVE=0
export SUPERLU_MAXSUP=256
export SUPERLU_RELAX=64
export SUPERLU_NUM_LOOKAHEADS=2
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

OUT_DIR=$DEMO_DIR/bench_superlu_scaling_$SLURM_JOB_ID
mkdir -p $OUT_DIR

# name nodes ranks_per_node nrow ncol omp_threads extra_worker_options
CONFIGS=(
    "n1_r4_t1      1   4  2  2  1"
    "n4_r16_t1     4   4  4  4  1"
    "n4_r16_t32    4   4  4  4 32"
    "n4_r64_t1     4  16  8  8  1"
    "n4_r64_t8     4  16  8  8  8"
    "n4_r256_t1    4  64 16 16  1"
    "n4_r64_t1_parsymb 4 16 8 8 1 -q 5 -s 1"
)

for cfg in "${CONFIGS[@]}"; do
    read -r name nnode rpn nrow ncol nth extra <<< "$cfg"
    nrank=$((nnode * rpn))
    cpt=$((256 / rpn))  # logical CPUs per rank (2 hyperthreads per core)
    export OMP_NUM_THREADS=$nth
    RUN_DIR=$PSCRATCH/gptune_superlu_scaling/$SLURM_JOB_ID/$name
    mkdir -p $RUN_DIR
    export CONTROL_FILE=$RUN_DIR/control.txt
    export DATA_FILE=$RUN_DIR/data.bin
    export RESULT_FILE=$RUN_DIR/result.bin
    rm -f $CONTROL_FILE $DATA_FILE $RESULT_FILE

    echo "=== $name: $nnode nodes, $nrank ranks ($nrow x $ncol), $nth threads, extra: ${extra:-none}  $(date +%T)"
    srun -N $nnode -n $nrank --ntasks-per-node=$rpn -c $cpt --cpu-bind=cores \
        python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c $ncol -r $nrow -d 1 -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 1 -n 0 $extra \
        > $OUT_DIR/${name}_worker.log 2>&1 &
    worker_pid=$!

    timeout 600 python -u bench_superlu_scaling.py > $OUT_DIR/${name}_driver.log 2>&1
    echo "driver exit status: $?"
    grep "^rep" $OUT_DIR/${name}_driver.log

    # stop the workers if the driver did not terminate them
    for ((i = 0; i < 60; ++i)); do kill -0 $worker_pid 2>/dev/null || break; sleep 1; done
    if kill -0 $worker_pid 2>/dev/null; then
        echo "workers still running, killing them"
        kill $worker_pid; wait $worker_pid
    fi
done
echo "=== done $(date +%T)"
