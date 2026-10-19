#!/bin/bash -l
#SBATCH --account=m4293_g
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=bpack-logdet-sign
#SBATCH --output=logdet_sign_repro_%j.out

# ButterflyPACK H2 (GPU) determinant-sign reproducer: logdet_sign_repro.py on the driver (node 0) and
# ButterflyPACK's Python workers (dPy_BPACK_worker.py), one MPI rank per A100.
# Usage, from this directory:  sbatch run_logdet_sign_repro.sbatch
#        SIZES=20000,100000,200000,400000 sbatch run_logdet_sign_repro.sbatch
# Environment: SIZES (default 20000,100000), NMPI (16), RANKS_PER_NODE (4), NMIN_LEAF (8),
#   REDUCTION_THRESHOLD (4), VERBOSITY (1: the worker log has the factor-time "logdet:" line),
#   BPACK_DIR (the ButterflyPACK checkout whose build_gpu/lib is used).
# Output: logdet_sign_repro_<job>.out (driver: one RESULT line per size) and
#   logdet_sign_repro_<job>_worker.log (workers).

REPRO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
BPACK_DIR=${BPACK_DIR:-/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/ButterflyPACK/ButterflyPACK}
SIZES=${SIZES:-20000,100000}
NMPI=${NMPI:-16}
RANKS_PER_NODE=${RANKS_PER_NODE:-4}
NODES=$(( (NMPI + RANKS_PER_NODE - 1) / RANKS_PER_NODE ))
THREADS_PER_RANK=$((128 / RANKS_PER_NODE))
export OMP_NUM_THREADS=$((64 / RANKS_PER_NODE))

# the runtime environment of the GPTune H2 runs (run_env.sh, Perlmutter, Cray MPICH + GNU)
module load python/3.13 PrgEnv-gnu gcc-native/14 cray-libsci/26.03.0 cray-mpich/9.1.0 >/dev/null 2>&1
module load cudatoolkit craype-accel-nvidia80 >/dev/null 2>&1
export PYTHONPATH=$BPACK_DIR/build_gpu/lib:${PYTHONUSERBASE}/lib/python3.13/site-packages  # (mpi4py from the user site)
export BPACK_PYTHON_LIB_PATH=$BPACK_DIR/build_gpu/lib
export LD_LIBRARY_PATH=/global/cfs/cdirs/m2957/lib/magma_v2.10.0/lib:$LD_LIBRARY_PATH
export MPI4PY_RC_RECV_MPROBE=false MPICH_VERSION_DISPLAY=0 MPICH_SINGLE_HOST_ENABLED=0
BPACK_SEQUENTIAL_OPENBLAS=/global/cfs/cdirs/m2957/lib/lib/PrgEnv-gnu/OpenBLAS_sequential/build/install/lib/libopenblas.so.0

# the driver <-> worker files of the Python interface
SHM_DIR=/dev/shm/bpack_logdet_repro_${SLURM_JOB_ID:-$$}
mkdir -p $SHM_DIR
export CONTROL_FILE=$SHM_DIR/control.txt DATA_FILE=$SHM_DIR/data.bin RESULT_FILE=$SHM_DIR/result.bin MAX_ID_FILE=10

cd $REPRO_DIR
echo "ButterflyPACK $(git -C $BPACK_DIR log --oneline -1) | sizes $SIZES | $NMPI ranks on $NODES nodes | nmin_leaf ${NMIN_LEAF:-8}"

WORKER_OPTS=(--xyzsort 0 --format 7 --sym 1 --reduction_threshold ${REDUCTION_THRESHOLD:-4} --tol_comp 1e-10
             --h2_id_proxy 2 --baca_batch 32 --h2_id_radius 2 --nmin_leaf ${NMIN_LEAF:-8} --errsol 0
             --verbosity ${VERBOSITY:-1} --h2_unstructured 1 --H2_use_gpu 1
             --H2_XRR_factor 1 --h2_lazy_schur 2 --h2_use_sketch 2)
srun -N $NODES -n $NMPI -c $THREADS_PER_RANK --cpu_bind=cores --ntasks-per-node=$RANKS_PER_NODE --gpus-per-node=4 \
    env LD_PRELOAD="$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_cuda.so:$BPACK_SEQUENTIAL_OPENBLAS" \
        OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=1 MPICH_ASYNC_PROGRESS=1 \
    bash -c 'export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4)); exec "$@"' worker \
    python -u $BPACK_DIR/build_gpu/lib/dPy_BPACK_worker.py -option "${WORKER_OPTS[@]}" \
    > logdet_sign_repro_${SLURM_JOB_ID}_worker.log 2>&1 &
WORKER_PID=$!

OMP_NUM_THREADS=32 python -u logdet_sign_repro.py --sizes $SIZES &
DRIVER_PID=$!
# the driver would wait forever for workers that have exited
while kill -0 $DRIVER_PID 2>/dev/null; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "the workers exited before the driver finished; see logdet_sign_repro_${SLURM_JOB_ID}_worker.log"
        kill $DRIVER_PID
        break
    fi
    sleep 5
done
wait $DRIVER_PID
wait $WORKER_PID
echo "factor-time logdet lines of the workers:"
grep "^logdet:" logdet_sign_repro_${SLURM_JOB_ID}_worker.log
rm -rf $SHM_DIR
