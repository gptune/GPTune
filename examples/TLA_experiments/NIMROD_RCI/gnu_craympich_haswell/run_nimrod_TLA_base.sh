#!/bin/bash

cd ../../../../

export machine=cori
export proc=haswell   # knl,haswell
export mpi=craympich  # openmpi,craympich
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

module load python/$PY_VERSION-anaconda-$PY_TIME
module swap PrgEnv-intel PrgEnv-gnu
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

seed=881
for expid in 0
do  
    seed=$( expr ${seed} + ${expid} )
    nstep=30
    nrun=200
    npilot=200

    #rm -rf gptune.db/*.json # do not load any database 
    bash nimrod_TLA_base.sh -a $nstep -b $expid -c $seed -d $nrun -e $npilot | tee log.nimrod_TLA_base_nstep${nstep}_expid${expid}_seed${seed}_nrun${nrun}_npilot${npilot}  #a: nstepmax b: nstepmin c: Nloop d: optimization e: expid f: seed
    #cp gptune.db/NIMROD.json  gptune.db/NIMROD.json_$(timestamp)
done
