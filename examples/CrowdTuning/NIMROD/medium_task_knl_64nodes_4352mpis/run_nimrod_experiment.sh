#!/bin/bash

cd ../../../../

export machine=cori
export proc=knl       # knl,haswell
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
module unload darshan
module swap craype-haswell craype-mic-knl
module swap PrgEnv-intel PrgEnv-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
#export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/gptune/:$PYTHONPATH
#MPIRUN=mpirun
#software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
#loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")

export PYTHONPATH=$PWD/GPTune:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
# export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export GPTUNEROOT=$PWD

cd -

#expid=0 # for expid in 0 1 2 .. set the expid for each tuning batch
#seed=881 # seed=$( expr ${seed} + ${expid} ) .. set a different seed number for each tuning batch

for expid in 0 1 2
do
    seed=881
    seed=$( expr ${seed} + ${expid} )

    for tuning_method in SLA
    do
        for npilot in 0
        do
            nstep=30
            nrun=20

            bash nimrod_tuning.sh -a $nstep -b $expid -c $seed -d $nrun -e $npilot -f ${tuning_method} | tee log.nimrod_${tuning_method}_nstep${nstep}_expid${expid}_seed${seed}_nrun${nrun}_npilot${npilot}
        done
    done

    for tuning_method in default_parameter
    do
        for npilot in 1
        do
            nstep=30
            nrun=1

            bash nimrod_tuning.sh -a $nstep -b $expid -c $seed -d $nrun -e $npilot -f ${tuning_method} | tee log.nimrod_${tuning_method}_nstep${nstep}_expid${expid}_seed${seed}_nrun${nrun}_npilot${npilot}
        done
    done

    for tuning_method in TLA_LCM TLA_Regression TLA_LCM_BF TLA_Sum
    do
        for npilot in 0
        do
            nstep=30
            nrun=20

            bash nimrod_tuning.sh -a $nstep -b $expid -c $seed -d $nrun -e $npilot -f ${tuning_method} | tee log.nimrod_${tuning_method}_nstep${nstep}_expid${expid}_seed${seed}_nrun${nrun}_npilot${npilot}
        done
    done
done
