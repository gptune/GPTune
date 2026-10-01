#!/bin/bash
#SBATCH --account=m2957
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=40:00:00
#SBATCH --job-name=gp-scaling
#SBATCH --output=scaling_gpu_%j.out

# Scaling test of the sparse-GP hyperparameter training of model_comparison_updated_superlu.py on 4 GPU
# nodes (16 SuperLU ranks, see superlu_gpu_env.sh). For every sample count in SCALING_NS, in order:
# gradient and finite-difference L-BFGS, then random-walk MCMC, then MALA (whose sampling stops after 3x
# the shorter L-BFGS training time), each step in its own SuperLU worker session so that a failed step
# does not stop the others. All files go to RESULTS_DIR; plots: python plot_training_results.py RESULTS_DIR 4
# Usage: SCALING_NS="100000 200000" RESULTS_DIR=<dir> sbatch -D <dir> run_scaling_gpu.sbatch
# (SCALING_OPTIMIZERS="gradient,finite difference" runs only the L-BFGS step)

GPTUNE_ROOT=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack
DEMO_DIR=$GPTUNE_ROOT/examples/GPTune-Demo
OBJ_CHOICE=${OBJ_CHOICE:-4}
RESULTS_DIR=${RESULTS_DIR:-$DEMO_DIR/scaling_obj$OBJ_CHOICE}

. $DEMO_DIR/superlu_gpu_env.sh
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR
echo "SCALING_NS=$SCALING_NS RESULTS_DIR=$RESULTS_DIR"

# GPU utilization on the first node, sampled every 10 seconds
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 > scaling_gpu_${SLURM_JOB_ID}_gpu_node0.csv &
smi_pid=$!

# the steps run for every N, separated by ';' (default: all three)
IFS=';' read -ra optimizer_steps <<< "${SCALING_OPTIMIZERS:-gradient,finite difference;mcmc;mala}"
echo "SCALING_OPTIMIZERS: ${optimizer_steps[*]}"

for N in $SCALING_NS; do
    for optimizers in "${optimizer_steps[@]}"; do
        export GP_NS=$N GP_OPTIMIZERS="$optimizers"
        superlu_session scaling_gpu_${SLURM_JOB_ID}_N${N}_${optimizers//[, ]/_} || echo "step N=$N $optimizers failed"
    done
done
python $DEMO_DIR/plot_training_results.py $RESULTS_DIR $OBJ_CHOICE
kill $smi_pid
