#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
    cd $GPTUNEROOT/examples/STRUMPACK
    rm -rf gptune.db/*.json # do not load any database 
    tp=STRUMPACK_Poisson3d
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_Poisson3d.py -ntask 1 -nrun 10 -machine cori 

    cd $GPTUNEROOT/examples/STRUMPACK
    rm -rf gptune.db/*.json # do not load any database
    tp=STRUMPACK_KRR
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json 
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR.py -ntask 1 -nrun 10 -machine cori -npernode $cores 

    if [[ $ModuleEnv == *"gpu"* ]]; then
        cd $GPTUNEROOT/examples/STRUMPACK
        rm -rf gptune.db/*.json # do not load any database 
        tp=STRUMPACK_MMdoubleMPIDist_GPU
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_1gpu.py -npernode 1 -ntask 1 -nrun 20
    fi    
fi
