#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --job-name=gp-inla-wendland
#SBATCH --output=inla_wendland_gpu_%j.out

# Wendland vs INLA sparse-GP comparison (model_comparison_inla_wendland.py) with SuperLU_DIST workers on GPU
# nodes, as run_superlu_gpu.sbatch: 16 MPI ranks on a 4x4 process grid, 4 ranks per node, one A100 per rank.
# Usage: sbatch run_inla_wendland_gpu.sbatch            (objective 4: 2D Schwefel, N=102400, both models, 4 optimizers)
#        GP_NS=409600 GP_MODELS=inla GP_OPTIMIZERS="gradient,finite difference" sbatch run_inla_wendland_gpu.sbatch
# The run writes its files in RESULTS_DIR (default: inla_vs_wendland/<job id> in this directory), one subdirectory per model.

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
OBJ_CHOICE=${OBJ_CHOICE:-4}
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR/inla_vs_wendland/$SLURM_JOB_ID}
export GP_NS=${GP_NS:-102400}
export GP_MODELS=${GP_MODELS:-wendland,inla}
export GP_OPTIMIZERS=${GP_OPTIMIZERS:-gradient,finite difference,mcmc,mala}
DRIVER_SCRIPT=$DEMO_DIR/model_comparison_inla_wendland.py

. $DEMO_DIR/superlu_gpu_env.sh
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR
echo "GP_NS=$GP_NS GP_MODELS=$GP_MODELS GP_OPTIMIZERS=$GP_OPTIMIZERS GP_INLA_SHAPE=${GP_INLA_SHAPE:-sqrt(N)} RESULTS_DIR=$RESULTS_DIR"

LOG_PREFIX=inla_wendland_gpu_${SLURM_JOB_ID}
srun -N $NNODE -n $NCORE_VAL_TOT --ntasks-per-node=4 --gpus-per-task=1 \
    bash -c 'echo "rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' | sort -n -k2

# GPU utilization on the first node, sampled every 2 seconds
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > ${LOG_PREFIX}_gpu_node0.csv &
smi_pid=$!

superlu_session $LOG_PREFIX
kill $smi_pid
