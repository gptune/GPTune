#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=gp-kaust
#SBATCH --output=kaust_gpu_%j.out

# Sub-competition 2b, dataset 1 of the 2021 KAUST competition (900K training, 100K testing values), Wendland vs
# INLA sparse GPs (kaust_inla_wendland.py), with SuperLU_DIST workers on GPU nodes as run_superlu_gpu.sbatch:
# 16 MPI ranks on a 4x4 process grid, one A100 per rank.
# Usage: sbatch run_kaust_gpu.sbatch                          (both models, gradient and finite-difference L-BFGS)
#        GP_MODELS="inla:shape=1024,inla:shape=2048" GP_OPTIMIZERS="gradient" sbatch run_kaust_gpu.sbatch
# KAUST_DATA is the npz of the data (default $PSCRATCH/kaust2021/kaust2021_2b_dataset01.npz). The run writes its
# files in RESULTS_DIR (default: kaust_compare/<job id> in this directory), one subdirectory per model.

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
OBJ_CHOICE=kaust  # only logged: the driver reads the environment below, not the standard input
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR/kaust_compare/$SLURM_JOB_ID}
export KAUST_DATA=${KAUST_DATA:-$PSCRATCH/kaust2021/kaust2021_2b_dataset01.npz}
export GP_MODELS=${GP_MODELS:-wendland,inla}
export GP_OPTIMIZERS=${GP_OPTIMIZERS:-gradient,finite difference}
DRIVER_SCRIPT=$DEMO_DIR/kaust_inla_wendland.py

. $DEMO_DIR/superlu_gpu_env.sh
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR
echo "KAUST_DATA=$KAUST_DATA GP_MODELS=$GP_MODELS GP_OPTIMIZERS=$GP_OPTIMIZERS RESULTS_DIR=$RESULTS_DIR"

LOG_PREFIX=kaust_gpu_${SLURM_JOB_ID}
srun -N $NNODE -n $NCORE_VAL_TOT --ntasks-per-node=4 --gpus-per-task=1 \
    bash -c 'echo "rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' | sort -n -k2

# GPU utilization on the first node, sampled every 2 seconds
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > ${LOG_PREFIX}_gpu_node0.csv &
smi_pid=$!

superlu_session $LOG_PREFIX
kill $smi_pid
