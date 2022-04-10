#!/bin/bash

cd ../../../

export machine=cori
export proc=haswell   # knl,haswell
export mpi=openmpi    # openmpi,craympich
export compiler=gnu   # gnu, intel
export ModuleEnv=$machine-$proc-$mpi-$compiler

if [[ $NERSC_HOST = "cori" ]]; then
    # PY_VERSION=3.7
    # PY_TIME=2019.07
    # MKL_TIME=2019.3.199

    PY_VERSION=3.8
    PY_TIME=2020.11
    MKL_TIME=2020.2.254
fi


module load gcc/8.3.0
module unload cray-mpich
module unload openmpi
module unload PrgEnv-intel
module load PrgEnv-gnu
module load openmpi/4.0.1
module unload craype-hugepages2M
module unload cray-libsci
module unload atp    
module load python/$PY_VERSION-anaconda-$PY_TIME
export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
#export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/gptune/:$PYTHONPATH
export PYTHONPATH=$PWD/GPTune/:$PYTHONPATH
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

nrun=20
for npilot in 0
do
    for tvalue in 1.01 1.02 1.1 1.2
    do
        for nbatch in 0 1 2 3 4
        do
            for tuning_method in SLA TLA_Sum TLA_Regression TLA_LCM_GPY
            do
                echo "Run demo base tuning_method=${tuning_method} tvalue=${tvalue} nbatch=${nbatch}"
                ./demo_tuning.py -tuning_method ${tuning_method} -tvalue ${tvalue} -nbatch ${nbatch} -npilot ${npilot} -nrun ${nrun}
            done
        done
        echo "\n"
    done
done

