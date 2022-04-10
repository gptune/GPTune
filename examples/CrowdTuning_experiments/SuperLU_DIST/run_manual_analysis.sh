#!/bin/bash

export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/superlu_dist/parmetis-4.0.3/install/lib/
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages

mpirun --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_manual_analysis.py

#npilot=10
#for tvalue in 1.1 1.2 1.3 1.4 1.5 1.01 1.02 1.03 1.04 1.05
#do
#    for nbatch in {0..9}
#    do
#        echo "Run TLA RWS tvalue=${tvalue} nbatch=${nbatch}"
#        ./demo_TLA_RWS.py -npilot ${npilot} -tvalue ${tvalue} -nbatch ${nbatch}
#        mv models_weights.log gptune.db/RWS_models_weights_${npilot}pilot_${tvalue}_${nbatch}.log
#    done
#    echo "\n"
#done

#npilot=10
#for tvalue in 1.1 1.2 1.3 1.4 1.5 1.01 1.02 1.03 1.04 1.05
#do
#    for nbatch in {0..9}
#    do
#        echo "Run TLA WS tvalue=${tvalue} nbatch=${nbatch}"
#        ./demo_TLA_WS.py -npilot ${npilot} -tvalue ${tvalue} -nbatch ${nbatch}
#    done
#    echo "\n"
#done


