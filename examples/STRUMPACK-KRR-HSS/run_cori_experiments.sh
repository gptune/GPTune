#!/bin/bash

cd ../../
. run_env.sh
cd -

export PYTHONPATH=$PWD/STRUMPACK/install/include/python/:$PYTHONPATH

if [[ ${1} != "" ]]; then
    dataset=${1}
else
    dataset="susy_10Kn"
fi

if [[ ${2} != "" ]]; then
    nrun=${2}
else
    nrun=10
fi

if [[ ${3} != "" ]]; then
    optimization=${3}
else
    optimization=10
fi

rm -rf gptune.db/*.json # do not load any database
$MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR_HSS.py -dataset ${dataset} -ntask 1 -nrun ${nrun} -machine cori -npernode ${cores} -optimization ${optimization} | tee a.out.log
