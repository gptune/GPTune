#!/bin/bash -l

#SBATCH --account=m2957
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --constraint=cpu
#SBATCH -t 20:00:00
#SBATCH -J GPTune_H2
#SBATCH --mail-user=liuyangzhuan@lbl.gov

cd ../../
. run_env.sh
cd -


#MPI+OMP settings:
################################################# 
nmpi=64 # number of MPIs
NTH=8 # number of OMP threads
export OMP_NUM_THREADS=$NTH
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


#ButterflyPACK settings:
################################################# 
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/
export PYTHONPATH=$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export BPACK_SEQUENTIAL_OPENBLAS=$CFS/m2957/lib/lib/PrgEnv-gnu/OpenBLAS_sequential/build/install/lib/libopenblas.so.0
if [ ! -r "$BPACK_SEQUENTIAL_OPENBLAS" ]; then
    echo "Missing sequential OpenBLAS: $BPACK_SEQUENTIAL_OPENBLAS" >&2
    exit 1
fi
################################################# 


## The following sets the file names for the butterflypack file interface
################################################# 
export CONTROL_FILE="control.txt"  ## this file is used to pass flags and parameters between the master driver and butterflypack workers 
export DATA_FILE="data.bin" ## this file is used to pass covariance matrix and rhs from the master driver to butterflypack workers 
export RESULT_FILE="result.bin" ## this file is used to pass solution vector and logdet from butterflypack workers to the master driver 
export MAX_ID_FILE=10 ## this is the maximum number of BPACK instances 
#################################################


if [ $ModuleEnv = 'perlmutter-milan-craympich-gnu' ]; then
    CORES_PER_NODE=128
    THREADS_PER_RANK=`expr $NTH \* 2`								 
    NODE_VAL=`expr $nmpi / $CORES_PER_NODE \* $NTH`
    MPIARG="-N ${NODE_VAL} -c ${THREADS_PER_RANK} --cpu_bind=cores"
fi

############## sequentially call the python driver Test_python_master.py, but parallelly launching the workers dPy_BPACK_worker.py 
for fid in $(seq 0 "$MAX_ID_FILE"); do
    rm -rf "$CONTROL_FILE.$fid" "$DATA_FILE.$fid" "$RESULT_FILE.$fid"
done

env LD_PRELOAD="$BPACK_SEQUENTIAL_OPENBLAS${LD_PRELOAD:+:$LD_PRELOAD}" OPENBLAS_NUM_THREADS=1 \


# ####### HODLR
# format=1
# $MPIRUN $MPIARG -n $nmpi python -u ${BPACK_PYTHON_LIB_PATH}/dPy_BPACK_worker.py -option --xyzsort 1 --format ${format} --sym 1 --IR_HODLR 10 --tol_comp 1e-10 --jitter_factor 0 --lrlevel 0 --reclr_leaf 5 --baca_batch 16 --nmin_leaf 128 --errsol 0 --verbosity 0 --knn 0 2>&1 | tee a.out_seperatelaunch_worker_format${format} &

####### H2
format=7
$MPIRUN $MPIARG -n $nmpi python -u ${BPACK_PYTHON_LIB_PATH}/dPy_BPACK_worker.py -option --xyzsort 0 --format ${format} --sym 1 --reduction_threshold 4 --tol_comp 1e-11  --h2_id_proxy 0 --baca_batch 32 --h2_id_radius 2 --nmin_leaf 64 --errsol 0 --verbosity 0  2>&1 | tee a.out_seperatelaunch_worker_format${format} &

python -u model_comparison_updated_bpack.py -format $format | tee a.out_gptune_format$format
python -c "from dPy_BPACK_wrapper import *; bpack_terminate()"









