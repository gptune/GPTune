#!/bin/bash

cd ../../../

export machine=cori
export proc=haswell
export mpi=openmpi
export compiler=gnu
export ModuleEnv=$machine-$proc-$mpi-$compiler

nodes=8
cores=32
gpus=0

if [[ $NERSC_HOST = "cori" ]]; then
    PY_VERSION=3.8
    PY_TIME=2020.11
    MKL_TIME=2020.2.254
fi  

module load gcc/8.3.0
module unload cray-mpich
module unload openmpi
module unload PrgEnv-intel
module load PrgEnv-gnu
module load openmpi/4.1.2
module unload craype-hugepages2M
module unload cray-libsci
module unload atp    
module load python/$PY_VERSION-anaconda-$PY_TIME
export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-github/lib/
export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/gptune/:$PYTHONPATH

export MPIRUN=mpirun
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
# export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD

cd -

nrun=100
npilot=100
nprocmin_pernode=1

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

db_options=$(echo ",\"sync_crowd_repo\":\"no\",\"save_model\":\"no\"")
machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")

#tp=PDGEQRF
#app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
#mkdir .gptune
#echo "$app_json$machine_json$software_json$db_options}" | jq '.' > .gptune/meta.json
#
#bash scalapack_tuning.sh -a ${nrun} -b ${npilot} -c ${nprocmin_pernode} -d 6000 -e 6000 | tee log.pdgeqrf
#cp gptune.db/PDGEQRF.json  gptune.db/PDGEQRF.json_$(timestamp)

nrun=30
nbatch=0

for tuning_method in SLA TLA_Sum TLA_Regression TLA_LCM_BF TLA_LCM TLA_Stacking TLA_Ensemble_Toggling TLA_Ensemble_Peeking TLA_Ensemble_Prob TLA_Ensemble_ProbDyn
do
    for npilot in 0
    do
        tp=PDGEQRF_${tuning_method}_${nbatch}
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        mkdir .gptune
        echo "$app_json$machine_json$software_json$db_options}" | jq '.' > .gptune/meta.json
        bash scalapack_tuning.sh -a ${tuning_method} -b ${nrun} -c ${npilot} -d ${nbatch} -e ${nprocmin_pernode} -f 12000 -g 12000 | tee log.pdgeqrf_${tuning_method}_nbatch${nbatch}_nrun${nrun}_npilot${npilot}
    done
done
