#!/bin/bash
# The same-nu H2 runs of 26 Nov 2026 (paper Tables IV and VI): the H2 GP with the Matern kernel of the smoothness of
# the INLA row of each dataset (george's Matern1Kernel, nu = 1, and Matern2Kernel, nu = 2), with the settings of the
# existing H2 row of that dataset except the kernel and the ButterflyPACK build (1d3d6d1 for all); plus H2 with the
# true Matern-3/2 kernel and INLA with nu = 2 on the synthetic Matern-3/2 field on the sphere. Each run is one GPU
# interactive allocation (salloc runs its command on the login node, so the run script is started in an
# interactive step, see run_h2_interactive.sh); the runs given are done one after the other.
# Usage: [DRYRUN=1] run_same_nu_interactive.sh [-A account] run...   (DRYRUN: print the commands only)
#   sst_full_m1 sst_ea_m1 grf_rbf_m2 grf_matern32_m1 grf_matern32_m32 heaton_m1      (1-2 nodes)
#   kaust2021_m1 kaust2022_1b1_m1 kaust2023_2b2_m2 kaust2023_2b4_m2 inla_nu2_grf_matern32  (4 nodes)
DEMO_DIR=/global/cfs/cdirs/m2957/liuyangz/my_research/GPTune_hodlr_vs_superlu_vs_scalapack/examples/GPTune-Demo
cd $DEMO_DIR
ACCOUNT=m4293_g
if [ "$1" = "-A" ]; then ACCOUNT=$2; shift 2; fi
LOG=$DEMO_DIR/spatial_h2/interactive_reruns.log
BP=$DEMO_DIR/../ButterflyPACK/ButterflyPACK_1d3d6d1
SPHERE="DRIVER=kaust KAUST_VAR_POINTS=5000 GP_OPTIMIZERS=gradient H2_UNSTRUCTURED=1 REDUCTION_THRESHOLD=8 NMPI=8 BPACK_DIR=$BP"
SST="GP_DATASET=sphere_modis_sst KAUST_DATA=$PSCRATCH/sphere/sphere_modis_sst_20210701.npz NMIN_LEAF=16 GP_NOISE='1e-6 1 0.1074' GP_AMPLITUDE='1e-2 1e2 9.031' GP_LENGTHSCALE='2e-3 1 0.0485'"
SST_EA="GP_DATASET=sphere_modis_sst_ea KAUST_DATA=$PSCRATCH/sphere/sphere_modis_sst_20210701_equalarea.npz NMIN_LEAF=32 GP_NOISE='1e-6 1 0.0883' GP_AMPLITUDE='1e-2 1e2 8.609' GP_LENGTHSCALE='2e-3 1 0.0504'"
GRF_RBF="GP_DATASET=sphere_grf_rbf KAUST_DATA=$PSCRATCH/sphere/sphere_grf_rbf_l0.05_s1_t0.01_uniform320000.npz NMIN_LEAF=8 GP_LENGTHSCALE='2e-3 1 0.1'"
GRF_M32="GP_DATASET=sphere_grf_matern32 KAUST_DATA=$PSCRATCH/sphere/sphere_grf_matern32_l0.05_s1_t0.01_uniform320000.npz NMIN_LEAF=8 GP_LENGTHSCALE='2e-3 1 0.1'"
K2023="DRIVER=kaust KAUST_VAR_POINTS=5000 GP_OPTIMIZERS=gradient H2_UNSTRUCTURED=1 NMIN_LEAF=16 VERBOSITY=1 BPACK_DIR=$BP GP_MODELS=h2:kern=Matern2"

run_one() {
    local run=$1 nodes script settings constraint=gpu time=01:00:00
    case $run in
        # sphere (Table VI): the settings of the H2 rows (59985120; 62046218/54/562), unscaled objective (norm=0)
        sst_full_m1)      nodes=2; script=run_spatial_h2_gpu.sbatch; constraint='gpu&hbm80g'; settings="$SPHERE $SST GP_MODELS=h2:iso=1:norm=0:kern=Matern1" ;;
        sst_ea_m1)        nodes=2; script=run_spatial_h2_gpu.sbatch; constraint='gpu&hbm80g'; settings="$SPHERE $SST_EA GP_MODELS=h2:iso=1:norm=0:kern=Matern1" ;;
        grf_rbf_m2)       nodes=2; script=run_spatial_h2_gpu.sbatch; constraint='gpu&hbm80g'; settings="$SPHERE $GRF_RBF GP_MODELS=h2:iso=1:norm=0:kern=Matern2" ;;
        grf_matern32_m1)  nodes=2; script=run_spatial_h2_gpu.sbatch; constraint='gpu&hbm80g'; settings="$SPHERE $GRF_M32 GP_MODELS=h2:iso=1:norm=0:kern=Matern1" ;;
        grf_matern32_m32) nodes=2; script=run_spatial_h2_gpu.sbatch; constraint='gpu&hbm80g'; settings="$SPHERE $GRF_M32 GP_MODELS=h2:iso=1:norm=0:kern=Matern32" ;;
        # Heaton (Table IV): the H2 row's run (59225251), gradient L-BFGS only
        heaton_m1)        nodes=1; script=run_spatial_h2_gpu.sbatch; settings="DRIVER=heaton H2_UNSTRUCTURED=1 NMPI=4 GP_MODELS=h2 GP_OPTIMIZERS=gradient HEATON_H2_KERN=Matern1 BPACK_DIR=$BP" ;;
        # KAUST 2021 (Table IV): the Matern-3/2 row's run (61195874), noise variance bounded below by 1e-8
        kaust2021_m1)     nodes=4; script=run_spatial_h2_gpu.sbatch; time=02:00:00; settings="DRIVER=kaust GP_OPTIMIZERS=gradient GP_DATASET=kaust_2b1 GP_MODELS=h2:kern=Matern1:noisemin=1e-8 BPACK_DIR=$BP" ;;
        # KAUST 2022 1b-1 replica (Table IV): the H2 row's run (60838155)
        kaust2022_1b1_m1) nodes=4; script=run_spatial_h2_gpu.sbatch; time=02:00:00; settings="DRIVER=kaust GP_OPTIMIZERS=gradient GP_DATASET=kaust2022_1b1 KAUST_DATA=$PSCRATCH/kaust2022/kaust2022_1b1_replica.npz GP_MODELS=h2:kern=Matern1 BPACK_DIR=$BP" ;;
        # KAUST 2023 datasets 2 and 4 (Table IV): the H2 rows' runs (60838156/57), started from INLA's nu = 2 fit
        kaust2023_2b2_m2) nodes=4; script=run_spatial_h2_gpu.sbatch; time=02:00:00; settings="$K2023 GP_DATASET=kaust2023_2b2 KAUST_DATA=$PSCRATCH/kaust2023/kaust2023_2b_dataset02.npz GP_NOISE='1e-6 1 0.2701' GP_AMPLITUDE='1e-2 1e2 1.931' GP_LENGTHSCALE='2e-3 1 0.137'" ;;
        kaust2023_2b4_m2) nodes=4; script=run_spatial_h2_gpu.sbatch; time=02:00:00; settings="$K2023 GP_DATASET=kaust2023_2b4 KAUST_DATA=$PSCRATCH/kaust2023/kaust2023_2b_dataset04.npz GP_NOISE='1e-6 1 0.2702' GP_AMPLITUDE='1e-2 1e2 2.424' GP_LENGTHSCALE='2e-3 1 0.133'" ;;
        # INLA nu = 2 on the Matern-3/2 field (Table VI): the settings of the nu = 1 row (59295240)
        inla_nu2_grf_matern32) nodes=4; script=run_kaust_gpu.sbatch; settings="GP_DATASET=sphere_grf_matern32 KAUST_DATA=$PSCRATCH/sphere/sphere_grf_matern32_l0.05_s1_t0.01_uniform320000.npz GP_MODELS=inla:sphere=8:nu=2 KAUST_VAR_POINTS=5000 GP_OPTIMIZERS=gradient" ;;
        *) echo "unknown run $run" >&2; return 1 ;;
    esac
    local out=$([ $script = run_kaust_gpu.sbatch ] && echo kaust_gpu || echo spatial_h2_gpu)
    local step='echo "$(date "+%F %T") '$run': job $SLURM_JOB_ID, interactive step on $(hostname)" >> '$LOG'; env '"$settings"' bash '$script' > '$out'_${SLURM_JOB_ID}.out 2>&1; rc=$?; echo "$(date "+%F %T") '$run': job $SLURM_JOB_ID done, exit $rc" >> '$LOG
    if [ -n "$DRYRUN" ]; then
        echo "salloc -A $ACCOUNT -q interactive -C '$constraint' -N $nodes --gpus-per-node=4 -t $time -J gp-same-nu srun --interactive --preserve-env -N 1 -n 1 bash -c"
        echo "    $step"
        return
    fi
    echo "$(date '+%F %T') $run: account $ACCOUNT, $nodes nodes, $script, $settings" >> $LOG
    salloc -A $ACCOUNT -q interactive -C "$constraint" -N $nodes --gpus-per-node=4 -t $time -J gp-same-nu \
        srun --interactive --preserve-env -N 1 -n 1 bash -c "$step"
}

for run in "$@"; do
    run_one $run || echo "$(date '+%F %T') $run: salloc failed" >> $LOG
done
