#!/bin/bash
cd ../../
. run_env.sh
cd -

# the following examples only work with openmpi
if [[ $ModuleEnv == *"openmpi"* ]]; then
    cd $GPTUNEROOT/examples/GPTune-Demo
    tp=GPTune-Demo
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json

    rm -rf gptune.db/*.json # do not load any database 
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./demo.py


    # Nloop=2
    # ntask=1
    # plot=0
    # restart=1
    # expid='0'
    # # for tuner in GPTune GPTuneBand hpbandster TPE
    # for tuner in GPTuneBand
    # do
    # rm -rf gptune.db/*.json # do not load any database 
    # $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

done


fi