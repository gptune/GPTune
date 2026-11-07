#!/bin/bash -l
#SBATCH --account=m2957
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=inla-sphere-check
#SBATCH --output=test_inla_sphere_%j.out

# The checks of the INLA/SPDE model on the sphere (test_inla_sphere.py) with one SuperLU_DIST worker through
# pdbridge (as test_GP_inla.sh for the lattice model). Submit from examples/GPTune-Demo.

DEMO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
cd $DEMO_DIR/../../
. run_env.sh
cd $DEMO_DIR
export OMP_NUM_THREADS=4
export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=0
export SUPERLU_ACC_SOLVE=0
RUN_DIR=$DEMO_DIR/inla_sphere_check/$SLURM_JOB_ID
mkdir -p $RUN_DIR
cd $RUN_DIR
export CONTROL_FILE="$RUN_DIR/control.txt"
export DATA_FILE="$RUN_DIR/data.bin"
export RESULT_FILE="$RUN_DIR/result.bin"

srun -N 1 -n 1 -c 8 python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c 1 -r 1 -d 1 -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 0 -n 0 > worker.log 2>&1 &
python -u $DEMO_DIR/test_inla_sphere.py -superlu 1 2>&1 | tee checks.log
python -c "from pdbridge import *; superlu_terminate()"
wait
echo "pdbridge refactorizations (pattern reused): $(grep -c 'refactor' worker.log)"
