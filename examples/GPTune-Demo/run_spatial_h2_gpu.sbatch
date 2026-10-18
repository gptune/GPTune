#!/bin/bash -l
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=gp-spatial-h2
#SBATCH --output=spatial_h2_gpu_%j.out

# The spatial datasets of kaust_inla_wendland.py (KAUST competitions) and heaton_inla_wendland.py (Heaton MODIS)
# with the dense RBF GP of george compressed in butterflypack's H2 format on GPUs (GP_MODELS=h2), with the H2 GPU
# settings of test_GP_h2_unstructured.sh: case aniso (one length scale per dimension, structured H2 of the
# original points), tol_comp 1e-10 with h2_id_proxy 2, one MPI rank per A100, a power of 4 ranks in 2D.
# Usage: sbatch run_spatial_h2_gpu.sbatch                                   (KAUST 2021 2b-1, 16 ranks on 4 nodes)
#        GP_DATASET=kaust2022_1b1 KAUST_DATA=$PSCRATCH/kaust2022/kaust2022_1b1_replica.npz sbatch run_spatial_h2_gpu.sbatch
#        DRIVER=heaton H2_UNSTRUCTURED=1 NMPI=4 GP_OPTIMIZERS="gradient,finite difference,mcmc,mala" sbatch -N 1 run_spatial_h2_gpu.sbatch
# The driver's environment (GP_MODELS, GP_OPTIMIZERS, KAUST_DATA, GP_DATASET, GP_NOISE, ...) is passed through.
# Results in RESULTS_DIR (default spatial_h2/<job id> in this directory).

DEMO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
DRIVER=${DRIVER:-kaust}
case $DRIVER in
    kaust)  DRIVER_SCRIPT=$DEMO_DIR/kaust_inla_wendland.py ;;
    heaton) DRIVER_SCRIPT=$DEMO_DIR/heaton_inla_wendland.py ;;
    gradcheck) DRIVER_SCRIPT=$DEMO_DIR/h2_gradient_check.py ;;  # GC_* settings, see the script
    *) echo "unknown DRIVER $DRIVER" >&2; exit 1 ;;
esac
export GP_MODELS=${GP_MODELS:-h2}
export GP_OPTIMIZERS=${GP_OPTIMIZERS:-gradient}
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR/spatial_h2/$SLURM_JOB_ID}
export GP_OUTPUT_DIR=$RESULTS_DIR
TOL_COMP=${TOL_COMP:-1e-10}
ID_PROXY=${ID_PROXY:-2}
# structured H2 (0) needs points in every box near a box (it stops on the Heaton data with cloud gaps); the
# unstructured H2 (1) works with the occupied boxes only
H2_UNSTRUCTURED=${H2_UNSTRUCTURED:-0}
# 3D points (e.g. on the sphere): REDUCTION_THRESHOLD=8 and a power of 8 ranks (NMPI=8)
# without H2_XRR_factor/h2_lazy_schur/h2_use_sketch the GPU box path (and the GPU solve) stays off, as in
# test_GP_h2_unstructured.sh
H2_OPTS=${H2_OPTS:---H2_XRR_factor 1 --h2_lazy_schur 2 --h2_use_sketch 2}

cd $DEMO_DIR/../../
. run_env.sh
cd $DEMO_DIR

nmpi=${NMPI:-16}
RANKS_PER_NODE=${RANKS_PER_NODE:-4}
NTH=$((64 / RANKS_PER_NODE))
THREADS_PER_RANK=$((128 / RANKS_PER_NODE))
NODE_VAL=$(( (nmpi + RANKS_PER_NODE - 1) / RANKS_PER_NODE ))
export OMP_NUM_THREADS=$NTH

# pdbridge is imported by the driver (for its sparse models), not used by the h2 runs
export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build_gpu/lib/
export PYTHONPATH=$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export BPACK_SEQUENTIAL_OPENBLAS=$CFS/m2957/lib/lib/PrgEnv-gnu/OpenBLAS_sequential/build/install/lib/libopenblas.so.0
if [ ! -r "$BPACK_SEQUENTIAL_OPENBLAS" ]; then
    echo "Missing sequential OpenBLAS: $BPACK_SEQUENTIAL_OPENBLAS" >&2
    exit 1
fi
SHM_DIR=/dev/shm/gptune_bpack_${SLURM_JOB_ID:-$$}
mkdir -p $SHM_DIR
export CONTROL_FILE="$SHM_DIR/control.txt"
export DATA_FILE="$SHM_DIR/data.bin"
export RESULT_FILE="$SHM_DIR/result.bin"
export MAX_ID_FILE=10

mkdir -p $RESULTS_DIR
cd $RESULTS_DIR
echo "driver=$DRIVER_SCRIPT GP_MODELS=$GP_MODELS GP_OPTIMIZERS=$GP_OPTIMIZERS GP_DATASET=${GP_DATASET:-} KAUST_DATA=${KAUST_DATA:-} format=7 tol_comp=$TOL_COMP id_proxy=$ID_PROXY h2_unstructured=$H2_UNSTRUCTURED nmin_leaf=${NMIN_LEAF:-64} h2_opts=$H2_OPTS nodes=$NODE_VAL nmpi=$nmpi threads=$NTH results=$RESULTS_DIR"
git -C $GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK log --oneline -1

module load cudatoolkit craype-accel-nvidia80 >/dev/null 2>&1
export LD_LIBRARY_PATH=/global/cfs/cdirs/m2957/lib/magma_v2.10.0/lib:$LD_LIBRARY_PATH
SRUN_ARGS=(-N ${NODE_VAL} -n $nmpi -c ${THREADS_PER_RANK} --cpu_bind=cores --ntasks-per-node=$RANKS_PER_NODE --gpus-per-node=4)
WORKER_ENV=(env LD_PRELOAD="$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_cuda.so:$BPACK_SEQUENTIAL_OPENBLAS${LD_PRELOAD:+:$LD_PRELOAD}"
            OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=1 MPICH_ASYNC_PROGRESS=1
            ${BPACK_GPU_HEAP_FRACTION:+BPACK_GPU_HEAP_FRACTION=$BPACK_GPU_HEAP_FRACTION})
GPU_WRAP=(bash -c 'export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % ${SLURM_GPUS_ON_NODE:-4})); exec "$@"' h2)
WORKER_OPTS=(--xyzsort 0 --format 7 --sym 1 --reduction_threshold ${REDUCTION_THRESHOLD:-4} --tol_comp ${TOL_COMP} --h2_id_proxy ${ID_PROXY}
             --baca_batch 32 --h2_id_radius 2 --nmin_leaf ${NMIN_LEAF:-64} --errsol 0 --verbosity ${VERBOSITY:-0} --h2_unstructured ${H2_UNSTRUCTURED} --H2_use_gpu 1)
LOG_PREFIX=spatial_h2_gpu_${SLURM_JOB_ID}
srun "${SRUN_ARGS[@]}" "${WORKER_ENV[@]}" "${GPU_WRAP[@]}" \
    python -u ${BPACK_PYTHON_LIB_PATH}/dPy_BPACK_worker.py -option "${WORKER_OPTS[@]}" ${H2_OPTS} \
    > ${LOG_PREFIX}_worker.log 2>&1 &
WORKER_PID=$!

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > ${LOG_PREFIX}_gpu_node0.csv &
SMI_PID=$!

OMP_NUM_THREADS=${DRIVER_OMP_THREADS:-32} python -u $DRIVER_SCRIPT > ${LOG_PREFIX}_driver.log 2>&1 &
DRIVER_PID=$!
# the driver would wait forever for workers that have exited, so stop it in that case
while kill -0 $DRIVER_PID 2>/dev/null; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "butterflypack workers exited before the driver finished; see ${LOG_PREFIX}_worker.log"
        kill $DRIVER_PID
        break
    fi
    sleep 10
done
wait $DRIVER_PID
echo "driver exit code: $?"
python -c "from dPy_BPACK_wrapper import *; bpack_terminate()"
kill $SMI_PID
wait
rm -rf $SHM_DIR
