#!/bin/bash
# Checks of the INLA/SPDE model of george (test_inla_lattice.py): the checks needing no SuperLU_DIST
# (log|Q| and the Matern marginals), then the ones with SuperLU_DIST workers (log|Q_c|, likelihood and
# predictions against dense references). Run in an allocation (srun), e.g. salloc -N 1 -C cpu -q interactive

cd ../../
. run_env.sh
cd -

NROW=1   # number of MPI row processes
NCOL=1   # number of MPI column processes
export OMP_NUM_THREADS=1

export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=0
export SUPERLU_ACC_SOLVE=0

# the file interface between the driver and the SuperLU_DIST workers
export CONTROL_FILE="control_inla.txt"
export DATA_FILE="data_inla.bin"
export RESULT_FILE="result_inla.bin"
rm -f $CONTROL_FILE $DATA_FILE $RESULT_FILE

python -u test_inla_lattice.py | tee a.out_inla_nosuperlu

$MPIRUN -n $((NROW * NCOL)) python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c $NCOL -r $NROW -d 1 -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 0 -n 0 | tee a.out_inla_worker &
python -u test_inla_lattice.py -superlu 1 | tee a.out_inla_superlu
python -c "from pdbridge import *; superlu_terminate()"
