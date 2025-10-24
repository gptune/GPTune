#!/bin/bash

cd ../../
. run_env.sh
cd -


#MPI settings:
################################################# 
NROW=2   # number of MPI row processes 
NCOL=2   # number of MPI column processes 
NPZ=1    # number of 2D process grids  
NTH=8 # number of OMP threads
################################################# 


#SUPERLU settings:
################################################# 
export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=0 # whether to do CPU or GPU numerical factorization
export GPU3DVERSION=0 # whether to do use the latest C++ numerical factorization 
export SUPERLU_ACC_SOLVE=0 # whether to do CPU or GPU triangular solve
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU
export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=100000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
nmpipergpu=1
export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # nmpipergpu>1 can better saturate GPU for some smaller matrices
################################################# 




NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`
export OMP_NUM_THREADS=$NTH
################################################# 


## The following sets the file names for the superlu file interface
################################################# 
export CONTROL_FILE="control.txt"  ## this file is used to pass flags and parameters between the master driver and superlu_dist workers 
export DATA_FILE="data.bin" ## this file is used to pass covariance matrix and rhs from the master driver to superlu_dist workers 
export RESULT_FILE="result.bin" ## this file is used to pass solution vector and logdet from superlu_dist workers to the master driver 
#################################################


############## sequentially call the python driver pddrive_master.py, but parallelly launching the workers pddrive_worker.py 
rm -rf $CONTROL_FILE
rm -rf $DATA_FILE
rm -rf $RESULT_FILE
$MPIRUN $MPIARG -n $NCORE_VAL_TOT python -u ${SUPERLU_PYTHON_LIB_PATH}/pddrive_worker.py -c $NCOL -r $NROW -d $NPZ -s 0 -q 4 -m 1 -p 1 -i 0 -b 0 -t 0 -n 0 | tee a.out_seperatelaunch_worker  &
# python -u ${SUPERLU_PYTHON_LIB_PATH}/GP_demo.py | tee a.out_seperatelaunch_master 

python -u model_comparison_updated_superlu.py | tee a.out_gptune
python -c "from pdbridge import *; superlu_terminate()"




