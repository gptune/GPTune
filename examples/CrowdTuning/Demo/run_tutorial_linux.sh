#!/bin/bash

cd ../../../

export machine=cleanlinux
export proc=unknown
export mpi=openmpi
export compiler=gnu
export nodes=1  # number of nodes to be used
export ModuleEnv=$machine-$proc-$mpi-$compiler

export OMPI_MCA_btl="^vader"  # disable vader, this causes runtime error when run in docker
MPIFromSource=1 # whether openmpi was built from source when installing GPTune

if [[ $MPIFromSource = 1 ]]; then
    export PATH=$PWD/openmpi-4.0.1/bin:$PATH
    export MPIRUN=$PWD/openmpi-4.0.1/bin/mpirun
    export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
else
    export PATH=$PATH
    export MPIRUN=
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    if [[ -z "$MPIRUN" ]]; then
        echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi has not been built from source when installing GPTune, please set MPIRUN, PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above."
        exit
    fi
fi

export PYTHONPATH=$PWD/build/gptune/:$PYTHONPATH
    export PATH=$PWD/env/bin/:$PATH
    export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/OpenBLAS:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/

cores=2
gpus=0

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
npilot=0
tvalue=1.0
echo "Run demo tutorial (automatic upload of function evaluations)"
./demo_tutorial.py -tvalue ${tvalue} -npilot ${npilot} -nrun ${nrun}

