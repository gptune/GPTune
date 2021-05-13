#!/bin/bash -l

module load python/3.7-anaconda-2019.10
module unload cray-mpich
module swap PrgEnv-intel PrgEnv-gnu
module load openmpi/4.0.1

cd ../../

export GPTUNEROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONWARNINGS=ignore

cd -

mkdir -p TLA_experiments

# tvalue 0.9 database file needs to be updated manually: tvalue round up issue (8.99999...) (TODO: deal with this automatically)
for optimization in {"GPTune","opentuner","hpbandster"}
do
#    for tvalue in {0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5}
    for tvalue in {0.6,0.8,1.0,1.2,1.4}
    do
        for nrun in {10,20,30,40,50}
        do
            rm -rf gptune.db
            mpirun -n 1 python ./demo.py -ntask 1 -nrun ${nrun} -tvalue ${tvalue} -optimization ${optimization} | tee a.out.log
            mkdir TLA_experiments/SLA-${optimization}-${tvalue}-${nrun}
            if [[ ${optimization} = 'GPTune' ]]; then
                mv gptune.db/GPTune-Demo.json TLA_experiments/SLA-${optimization}-${tvalue}-${nrun}/GPTune-Demo.json
            fi
            mv a.out.log TLA_experiments/SLA-${optimization}-${tvalue}-${nrun}/a.out.log
        done
    done
done

#for tvalue in {0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5}
for tvalue in {0.6,0.8,1.0,1.2,1.4}
do
    for nrun in {10,20,30,40,50}
    do
        for transfer_task in {0.6,0.8,1.0,1.2,1.4}
        do
            if [[ ${tvalue} != ${transfer_task} ]]; then
                rm -rf gptune.db
                cp -r TLA_experiments/SLA-GPTune-${transfer_task}-50 gptune.db
                mpirun -n 1 python ./demo_TLA.py -ntask 2 -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${transfer_task} | tee a.out.log
                mkdir TLA_experiments/TLA-${tvalue}-${transfer_task}-${nrun}
                mv gptune.db/GPTune-Demo.json TLA_experiments/TLA-${tvalue}-${transfer_task}-${nrun}/GPTune-Demo.json
                mv a.out.log TLA_experiments/TLA-${tvalue}-${transfer_task}-${nrun}/a.out.log
            fi
        done
    done
done

for nrun in {10,20,30,40,50}
do
    tvalue=0.8
    tvalue2=0.6
    tvalue3=1.0

    rm -rf gptune.db
    rm db.out
    ./merge_db.py TLA_experiments/SLA-GPTune-${tvalue2}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue3}-50/GPTune-Demo.json
    mkdir -p gptune.db
    mv db.out gptune.db/GPTune-Demo.json
    mpirun -n 1 python ./demo_TLA.py -ntask 3 -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${tvalue2} -tvalue3 ${tvalue3} | tee a.out.log
    mkdir TLA_experiments/TLA-${tvalue}-3tasks-${nrun}
    mv gptune.db/GPTune-Demo.json TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/GPTune-Demo.json
    mv a.out.log TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/a.out.log
done

for nrun in {10,20,30,40,50}
do
    tvalue=1.0
    tvalue2=0.8
    tvalue3=1.2

    rm -rf gptune.db
    rm db.out
    ./merge_db.py TLA_experiments/SLA-GPTune-${tvalue2}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue3}-50/GPTune-Demo.json
    mkdir -p gptune.db
    mv db.out gptune.db/GPTune-Demo.json
    mpirun -n 1 python ./demo_TLA.py -ntask 3 -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${tvalue2} -tvalue3 ${tvalue3} | tee a.out.log
    mkdir TLA_experiments/TLA-${tvalue}-3tasks-${nrun}
    mv gptune.db/GPTune-Demo.json TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/GPTune-Demo.json
    mv a.out.log TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/a.out.log
done

for nrun in {10,20,30,40,50}
do
    tvalue=1.2
    tvalue2=1.0
    tvalue3=1.4

    rm -rf gptune.db
    rm db.out
    ./merge_db.py TLA_experiments/SLA-GPTune-${tvalue2}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue3}-50/GPTune-Demo.json
    mkdir -p gptune.db
    mv db.out gptune.db/GPTune-Demo.json
    mpirun -n 1 python ./demo_TLA.py -ntask 3 -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${tvalue2} -tvalue3 ${tvalue3} | tee a.out.log
    mkdir TLA_experiments/TLA-${tvalue}-3tasks-${nrun}
    mv gptune.db/GPTune-Demo.json TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/GPTune-Demo.json
    mv a.out.log TLA_experiments/TLA-${tvalue}-3tasks-${nrun}/a.out.log
done

for nrun in {10,20,30,40,50}
do
    tvalue=1.0
    tvalue2=0.6
    tvalue3=0.8
    tvalue4=1.2
    tvalue5=1.4

    rm -rf gptune.db
    rm db.out
    ./merge_db.py TLA_experiments/SLA-GPTune-${tvalue2}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue3}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue4}-50/GPTune-Demo.json TLA_experiments/SLA-GPTune-${tvalue5}-50/GPTune-Demo.json
    mkdir -p gptune.db
    mv db.out gptune.db/GPTune-Demo.json
    mpirun -n 1 python ./demo_TLA.py -ntask 5 -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${tvalue2} -tvalue3 ${tvalue3} -tvalue4 ${tvalue4} -tvalue5 ${tvalue5} | tee a.out.log
    mkdir TLA_experiments/TLA-${tvalue}-5tasks-${nrun}
    mv gptune.db/GPTune-Demo.json TLA_experiments/TLA-${tvalue}-5tasks-${nrun}/GPTune-Demo.json
    mv a.out.log TLA_experiments/TLA-${tvalue}-5tasks-${nrun}/a.out.log
done

exit
