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
for npilot in 0
do
    for tvalue in 1.01 1.02 1.1 1.2
    do
        for nbatch in 0 1 2 3 4
        do
            for tuning_method in SLA TLA_Regression TLA_LCM_GPY TLA_Stacking TLA_Ensemble
            do
                echo "Run demo base tuning_method=${tuning_method} tvalue=${tvalue} nbatch=${nbatch}"
                ./demo_tuning.py -tuning_method ${tuning_method} -tvalue ${tvalue} -nbatch ${nbatch} -npilot ${npilot} -nrun ${nrun}
            done
        done
        echo "\n"
    done
done

