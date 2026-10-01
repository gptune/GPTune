#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=gp-heaton
#SBATCH --output=heaton_gpu_%j.out

# The Heaton et al. MODIS case study, Wendland vs INLA sparse GPs (heaton_inla_wendland.py), with SuperLU_DIST
# workers on GPU nodes as run_superlu_gpu.sbatch: 16 MPI ranks on a 4x4 process grid, one A100 per rank.
# Usage: sbatch run_heaton_gpu.sbatch                         (satellite data, both models, 4 optimizers)
#        HEATON_DATASET=simulated GP_MODELS=inla GP_OPTIMIZERS="gradient" sbatch run_heaton_gpu.sbatch
# HEATON_DATA is the Data directory of https://github.com/finnlindgren/heatoncomparison (default
# $PSCRATCH/heatoncomparison/Data). The run writes its files in RESULTS_DIR (default: heaton_compare/<job id>
# in this directory), one subdirectory per model.

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
OBJ_CHOICE=heaton  # only logged: the driver reads the environment below, not the standard input
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR/heaton_compare/$SLURM_JOB_ID}
export HEATON_DATA=${HEATON_DATA:-$PSCRATCH/heatoncomparison/Data}
export HEATON_DATASET=${HEATON_DATASET:-satellite}
export GP_MODELS=${GP_MODELS:-wendland,inla}
export GP_OPTIMIZERS=${GP_OPTIMIZERS:-gradient,finite difference,mcmc,mala}
DRIVER_SCRIPT=$DEMO_DIR/heaton_inla_wendland.py

. $DEMO_DIR/superlu_gpu_env.sh
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR
echo "HEATON_DATA=$HEATON_DATA HEATON_DATASET=$HEATON_DATASET GP_MODELS=$GP_MODELS GP_OPTIMIZERS=$GP_OPTIMIZERS RESULTS_DIR=$RESULTS_DIR"

LOG_PREFIX=heaton_gpu_${SLURM_JOB_ID}
srun -N $NNODE -n $NCORE_VAL_TOT --ntasks-per-node=4 --gpus-per-task=1 \
    bash -c 'echo "rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' | sort -n -k2

# GPU utilization on the first node, sampled every 2 seconds
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > ${LOG_PREFIX}_gpu_node0.csv &
smi_pid=$!

superlu_session $LOG_PREFIX
kill $smi_pid
