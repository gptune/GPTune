#!/bin/bash
cd ../../
. run_env.sh
cd -

# the following examples only work with openmpi
if [[ $ModuleEnv == *"openmpi"* ]]; then
    cd $GPTUNEROOT/examples/GPTune-Demo-MO
    tp=GPTune-Demo-MO
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json

    rm -rf gptune.db/*.json # do not load any database 
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./OSY_MO.py -nrun 400 -npilot 200



fi