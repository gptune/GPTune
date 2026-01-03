#!/bin/bash

cd ../../
. run_env.sh
cd -


#MPI+OMP settings:
################################################# 
nmpi=2 # number of MPIs
NTH=1 # number of OMP threads
export OMP_NUM_THREADS=$NTH
################################################# 


#ButterflyPACK settings:
################################################# 
export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
################################################# 


## The following sets the file names for the butterflypack file interface
################################################# 
export CONTROL_FILE="control.txt"  ## this file is used to pass flags and parameters between the master driver and butterflypack workers 
export DATA_FILE="data.bin" ## this file is used to pass covariance matrix and rhs from the master driver to butterflypack workers 
export RESULT_FILE="result.bin" ## this file is used to pass solution vector and logdet from butterflypack workers to the master driver 
export MAX_ID_FILE=10 ## this is the maximum number of BPACK instances 
#################################################




############## sequentially call the python driver Test_python_master.py, but parallelly launching the workers dPy_BPACK_worker.py 
for fid in $(seq 0 "$MAX_ID_FILE"); do
    rm -rf "$CONTROL_FILE.$fid" "$DATA_FILE.$fid" "$RESULT_FILE.$fid"
done
mpirun --allow-run-as-root -n $nmpi python -u ${BPACK_PYTHON_LIB_PATH}/dPy_BPACK_worker.py -option --xyzsort 1 --tol_comp 1e-10 --lrlevel 0 --reclr_leaf 5 --nmin_leaf 128 --errsol 1 | tee a.out_seperatelaunch_worker &
python -u model_comparison_updated_bpack.py | tee a.out_gptune
python -c "from dPy_BPACK_wrapper import *; bpack_terminate()"












