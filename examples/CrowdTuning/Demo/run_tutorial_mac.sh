#!/bin/zsh

cd ../../../

MPIFromSource=1 # whether openmpi was built from source when installing GPTune
if [[ $MPIFromSource = 1 ]]; then
    export PATH=$PWD/openmpi-4.0.1/bin:$PATH
    export MPIRUN="$PWD/openmpi-4.0.1/bin/mpirun"
    export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$DYLD_LIBRARY_PATH
else
    export MPIRUN=
    export PATH=$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    export LIBRARY_PATH=$LIBRARY_PATH  
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH    
    if [[ -z "$MPIRUN" ]]; then
		echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi has not been built from source when installing GPTune, please set MPIRUN, PATH, LD_LIBRARY_PATH, DYLD_LIBRARY_PATH for your OpenMPI build correctly above."
		exit
	fi       
fi    

export PYTHONPATH=$PWD/build/gptune/:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/build/
export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
export PATH=$PWD/env/bin/:$PATH

export SCALAPACK_LIB=$PWD/scalapack-2.1.0/build/install/lib/libscalapack.dylib
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$DYLD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH

#export PYTHONPATH=$PWD/GPTune/:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/build/
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

