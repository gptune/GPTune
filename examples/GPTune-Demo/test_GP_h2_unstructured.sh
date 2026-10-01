#!/bin/bash -l

#SBATCH --account=m2957
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --constraint=cpu
#SBATCH -t 04:00:00
#SBATCH -J GPTune_H2_unstr
#SBATCH -o GPTune_H2_unstr_%j.out

# Compare H2 (format 7) GP hyperparameter optimization with isotropic vs anisotropic RBF kernels.
# Submit from examples/GPTune-Demo:
#   sbatch test_GP_h2_unstructured.sh <case> [NS] [objtype]
#   (override the MPI count with NMPI=<n> sbatch -N <nodes> ..., 16 MPIs per node)
#   (hyperparameter optimizers, run one after the other: OPTIMIZER=gradient (default), or a comma-separated
#    list of gradient, "finite difference", mcmc, mala, e.g. OPTIMIZER="gradient,finite difference,mcmc,mala")
#   (results directory under h2_unstructured_cmp: RUN_NAME=<name>; default: built from the settings)
#   (GPU nodes: USE_GPU=1 [RANKS_PER_NODE=<n>, default 4], with the GPU build in build_gpu and
#    --H2_use_gpu 1; submit with sbatch -C gpu --gpus-per-node=4 -N <nodes>)
#   (extra butterflypack options: H2_OPTS="...", e.g. "--H2_XRR_factor 1 --h2_lazy_schur 2 --h2_use_sketch 2")
#   (H2 compression tolerance: TOL_COMP=<tol>, default 1e-11)
#   (noise variance range: NOISEVAR="<min> <max> <initial>", linear scale; default: the GPTune default)
#   (length scale range: LENGTHSCALE="<min> <max> <initial>", linear scale; default: the GPTune default)
#   case: iso           model_isotropic=True,  original points,          structured H2 (as test_GP_hodlr_vs_bpack.sh)
#         iso_unstr     model_isotropic=True,  original points,          --h2_unstructured 1
#         aniso         model_isotropic=False, original points,          structured H2
#         aniso_scaled  model_isotropic=False, points / length scales,   --h2_unstructured 1
# Each case runs in its own directory h2_unstructured_cmp/<case>_<grad|fd>_obj<objtype>_N<NS-1>, since the
# master and the butterflypack workers communicate through files in the working directory.

CASE=${1:-aniso_scaled}
NS=${2:-102401}
OBJTYPE=${3:-2}
OPTIMIZER=${OPTIMIZER:-gradient}
NOISEVAR=${NOISEVAR:-}
LENGTHSCALE=${LENGTHSCALE:-}
USE_GPU=${USE_GPU:-0}
H2_OPTS=${H2_OPTS:-}
TOL_COMP=${TOL_COMP:-1e-11} # H2 compression tolerance
# 3D trees need a reduction threshold of at least 8
REDUCTION_THRESHOLD=${REDUCTION_THRESHOLD:-$([ "$OBJTYPE" = 3 -o "$OBJTYPE" = 5 ] && echo 8 || echo 4)}

DEMO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
cd $DEMO_DIR/../../
. run_env.sh
cd $DEMO_DIR


#MPI+OMP settings:
#################################################
nmpi=${NMPI:-64} # number of MPIs, a power of 4 in 2D and of 8 in 3D; the H2 tree must have enough levels: nmpi <= 4^(num_levels-2) in 2D
if [ "$USE_GPU" = 1 ]; then
    # GPU nodes: 64 cores and 4 A100s per node, shared round-robin by the ranks of the node
    RANKS_PER_NODE=${RANKS_PER_NODE:-4}
    NTH=$((64 / RANKS_PER_NODE)) # number of OMP threads
    THREADS_PER_RANK=$((128 / RANKS_PER_NODE))
    NODE_VAL=$(( (nmpi + RANKS_PER_NODE - 1) / RANKS_PER_NODE ))
else
    NTH=8 # number of OMP threads
    CORES_PER_NODE=128
    THREADS_PER_RANK=`expr $NTH \* 2`
    NODE_VAL=`expr $nmpi \* $NTH / $CORES_PER_NODE`
fi
export OMP_NUM_THREADS=$NTH
#################################################


#SUPERLU settings (only needed for importing pdbridge in the driver):
#################################################
export SUPERLU_PYTHON_LIB_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/build/lib/PYTHON/
export PYTHONPATH=$SUPERLU_PYTHON_LIB_PATH:$PYTHONPATH
#################################################


#ButterflyPACK settings:
#################################################
export BPACK_PYTHON_LIB_PATH=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/$([ "$USE_GPU" = 1 ] && echo build_gpu || echo build)/lib/
export PYTHONPATH=$BPACK_PYTHON_LIB_PATH:$PYTHONPATH
export BPACK_SEQUENTIAL_OPENBLAS=$CFS/m2957/lib/lib/PrgEnv-gnu/OpenBLAS_sequential/build/install/lib/libopenblas.so.0
if [ ! -r "$BPACK_SEQUENTIAL_OPENBLAS" ]; then
    echo "Missing sequential OpenBLAS: $BPACK_SEQUENTIAL_OPENBLAS" >&2
    exit 1
fi
# the master and the workers exchange vectors through these files, read and written by the master and
# MPI rank 0 on the first node: node-local memory is much faster than CFS for the multi-vector solves
SHM_DIR=/dev/shm/gptune_bpack_${SLURM_JOB_ID:-$$}
mkdir -p $SHM_DIR
export CONTROL_FILE="$SHM_DIR/control.txt"
export DATA_FILE="$SHM_DIR/data.bin"
export RESULT_FILE="$SHM_DIR/result.bin"
export MAX_ID_FILE=10
#################################################


case $CASE in
    iso)          ISOTROPIC=1; SCALED=0; H2_UNSTRUCTURED=0 ;;
    iso_unstr)    ISOTROPIC=1; SCALED=0; H2_UNSTRUCTURED=1 ;;
    aniso_scaled) ISOTROPIC=0; SCALED=1; H2_UNSTRUCTURED=1 ;;
    aniso)        ISOTROPIC=0; SCALED=0; H2_UNSTRUCTURED=0 ;;
    *) echo "unknown case $CASE" >&2; exit 1 ;;
esac

OPT_TAG=$(echo "$OPTIMIZER" | sed 's/finite difference/fd/g; s/gradient/grad/g; s/,/-/g')
NOISE_TAG=${NOISEVAR:+_noise$(echo $NOISEVAR | cut -d' ' -f1)}
LS_TAG=${LENGTHSCALE:+_ls$(echo $LENGTHSCALE | cut -d' ' -f3)}
# runs sharing a RUN_NAME directory (different NS) keep separate logs
LOG_SUFFIX=${RUN_NAME:+_N$((NS - 1))}
RUN_DIR=$DEMO_DIR/h2_unstructured_cmp/${RUN_NAME:-${CASE}_${OPT_TAG}_obj${OBJTYPE}_N$((NS - 1))${NOISE_TAG}${LS_TAG}}
mkdir -p $RUN_DIR
cd $RUN_DIR
for fid in $(seq 0 "$MAX_ID_FILE"); do
    rm -rf "$CONTROL_FILE.$fid" "$DATA_FILE.$fid" "$RESULT_FILE.$fid"
done
echo "case=$CASE optimizer=$OPTIMIZER noisevariance=${NOISEVAR:-default} lengthscale=${LENGTHSCALE:-default} NS=$NS objtype=$OBJTYPE isotropic=$ISOTROPIC scaled_geometry=$SCALED h2_unstructured=$H2_UNSTRUCTURED nodes=$NODE_VAL nmpi=$nmpi threads=$NTH use_gpu=$USE_GPU tol_comp=$TOL_COMP h2_opts=$H2_OPTS"
git -C $GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK log --oneline -1


####### H2 workers (sequential BLAS inside the OpenMP-threaded H2 code)
format=7
SRUN_ARGS=(-N ${NODE_VAL} -n $nmpi -c ${THREADS_PER_RANK} --cpu_bind=cores)
WORKER_ENV=(env LD_PRELOAD="$BPACK_SEQUENTIAL_OPENBLAS${LD_PRELOAD:+:$LD_PRELOAD}" OPENBLAS_NUM_THREADS=1)
GPU_WRAP=()
H2_GPU_OPTS=""
if [ "$USE_GPU" = 1 ]; then
    module load cudatoolkit craype-accel-nvidia80 >/dev/null 2>&1
    export LD_LIBRARY_PATH=/global/cfs/cdirs/m2957/lib/magma_master/lib:$LD_LIBRARY_PATH
    SRUN_ARGS+=(--ntasks-per-node=$RANKS_PER_NODE --gpus-per-node=4)
    # CUDA-aware MPI for the GPU backend, only in the workers: mpi4py initializes MPI before
    # butterflypack is loaded, so the GPU transport layer of Cray MPICH is preloaded
    WORKER_ENV=(env LD_PRELOAD="$CRAY_MPICH_ROOTDIR/gtl/lib/libmpi_gtl_cuda.so:$BPACK_SEQUENTIAL_OPENBLAS${LD_PRELOAD:+:$LD_PRELOAD}"
                OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=1 MPICH_ASYNC_PROGRESS=1
                H2_GPU_HEAP_GB=${H2_GPU_HEAP_GB:-$( [ $(( (RANKS_PER_NODE + 3) / 4 )) -ge 4 ] && echo 6 || echo $(( 32 / ((RANKS_PER_NODE + 3) / 4) )) )}
                H2_GPU_MATVEC=${H2_GPU_MATVEC:-0} H2_GPU_KEEP_OPERATORS=${H2_GPU_KEEP_OPERATORS:-1})
    # (the H2 device heap of a rank, H2_GPU_HEAP_GB: 32 GiB of its 40 GB A100 over the ranks sharing it, but
    #  6 GiB with 4 or more ranks per GPU, whose CUDA contexts and MAGMA workspaces also need room)
    # (the multiplies by the compression-only dK/dtheta operators run on the host, H2_GPU_MATVEC=0: with 65
    #  right-hand sides the device matvec was 1.3-4.5x slower at N=102,400 and up to 25x at N=16,384; the
    #  factorization of K keeps its device solve data while those operators are built, H2_GPU_KEEP_OPERATORS=1)
    # each rank sees one GPU of its node
    GPU_WRAP=(bash -c 'export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % ${SLURM_GPUS_ON_NODE:-4})); exec "$@"' h2)
    H2_GPU_OPTS="--H2_use_gpu 1"
fi
srun "${SRUN_ARGS[@]}" "${WORKER_ENV[@]}" "${GPU_WRAP[@]}" \
    python -u ${BPACK_PYTHON_LIB_PATH}/dPy_BPACK_worker.py -option --xyzsort 0 --format ${format} --sym 1 --reduction_threshold ${REDUCTION_THRESHOLD} --tol_comp ${TOL_COMP} --h2_id_proxy 0 --baca_batch 32 --h2_id_radius 2 --nmin_leaf 64 --errsol 0 --verbosity 0 --h2_unstructured ${H2_UNSTRUCTURED} ${H2_OPTS} ${H2_GPU_OPTS} \
    > worker${LOG_SUFFIX}.log 2>&1 &
WORKER_PID=$!

python -u $DEMO_DIR/model_comparison_updated_bpack.py -format $format -objtype $OBJTYPE -NS $NS -isotropic $ISOTROPIC -bpack_scaled_geometry $SCALED -optimizer "$OPTIMIZER" ${NOISEVAR:+-noisevariance $NOISEVAR} ${LENGTHSCALE:+-lengthscale $LENGTHSCALE} > driver${LOG_SUFFIX}.log 2>&1 &
DRIVER_PID=$!
# the driver would wait forever for workers that have exited, so stop it in that case
while kill -0 $DRIVER_PID 2>/dev/null; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "butterflypack workers exited before the driver finished; see worker${LOG_SUFFIX}.log"
        kill $DRIVER_PID
        break
    fi
    sleep 10
done
wait $DRIVER_PID
echo "driver exit code: $?"
python -c "from dPy_BPACK_wrapper import *; bpack_terminate()"
wait
rm -rf $SHM_DIR
