#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=heaton-inla
#SBATCH --output=heaton_inla_%j.out

# The Heaton et al. MODIS case study with the INLA/SPDE model (model_kern='INLA') of Model_George,
# the precision matrices being factored by SuperLU_DIST workers (heaton_inla.py <-> pddrive_worker.py
# file interface, as model_comparison_updated_superlu.py).
# HEATON_DATA: the Data directory of https://github.com/finnlindgren/heatoncomparison (needs pyreadr).

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
HEATON_DATA=${HEATON_DATA:-$PSCRATCH/heatoncomparison/Data}
DATASET=${DATASET:-satellite}
NROW=${NROW:-4}
NCOL=${NCOL:-4}
NTH=${NTH:-1}

cd $GPTUNE_ROOT
. ./run_env.sh
cd $DEMO_DIR

export SUPERLU_PYTHON_LIB_PATH=$GPTUNE_ROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export BPACK_PYTHON_LIB_PATH=$GPTUNE_ROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=0
export SUPERLU_ACC_SOLVE=0
export SUPERLU_MAXSUP=256
export SUPERLU_RELAX=64
export SUPERLU_NUM_LOOKAHEADS=2
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

RUN_DIR=$PSCRATCH/gptune_heaton_inla/$SLURM_JOB_ID
mkdir -p $RUN_DIR
export CONTROL_FILE=$RUN_DIR/control.txt
export DATA_FILE=$RUN_DIR/data.bin
export RESULT_FILE=$RUN_DIR/result.bin

NRANK=$((NROW * NCOL))
CPT=$((256 / NRANK))  # logical CPUs per rank (2 hyperthreads per core)
# -q 4: METIS nested dissection ordering, -m 1: symmetric pattern (Q_c is structurally symmetric)
srun -N 1 -n $NRANK -c $CPT --cpu-bind=cores \
    python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c $NCOL -r $NROW -d 1 -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 1 -n 0 \
    > $RUN_DIR/worker.log 2>&1 &
worker_pid=$!

python -u heaton_inla.py -data_dir $HEATON_DATA -dataset $DATASET -out $RUN_DIR/heaton_inla_$DATASET "$@"
echo "driver exit status: $?"

# stop the workers if the driver did not terminate them
for ((i = 0; i < 60; ++i)); do kill -0 $worker_pid 2>/dev/null || break; sleep 1; done
if kill -0 $worker_pid 2>/dev/null; then
    echo "workers still running, killing them"
    kill $worker_pid; wait $worker_pid
fi
echo "outputs in $RUN_DIR"
