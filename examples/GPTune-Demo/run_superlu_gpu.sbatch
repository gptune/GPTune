#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --job-name=gp-superlu-gpu
#SBATCH --output=superlu_gpu_%j.out

# Sparse-GP model comparison (model_comparison_updated_superlu.py) with SuperLU_DIST
# workers on GPU nodes: 16 MPI ranks on a 4x4 process grid, 4 ranks per node, one A100 per rank.
# The driver runs on the first node; it talks to the workers through the CONTROL/DATA/RESULT files.
# Usage: sbatch run_superlu_gpu.sbatch            (objective 4: 2D Schwefel)
#        OBJ_CHOICE=2 sbatch run_superlu_gpu.sbatch
#        GP_NS=200000 GP_OPTIMIZERS="gradient,finite difference" RESULTS_DIR=<dir> sbatch -D <dir> run_superlu_gpu.sbatch
# GP_NS and GP_OPTIMIZERS select the sample counts and optimizers (see model_comparison_updated_superlu.py);
# the run writes its files in RESULTS_DIR (default: this directory).

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
OBJ_CHOICE=${OBJ_CHOICE:-4}
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR}

. $DEMO_DIR/superlu_gpu_env.sh
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

LOG_PREFIX=superlu_gpu_${SLURM_JOB_ID}
srun -N $NNODE -n $NCORE_VAL_TOT --ntasks-per-node=4 --gpus-per-task=1 \
    bash -c 'echo "rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' | sort -n -k2

# GPU utilization on the first node, sampled every 2 seconds
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > ${LOG_PREFIX}_gpu_node0.csv &
smi_pid=$!

superlu_session $LOG_PREFIX
kill $smi_pid
