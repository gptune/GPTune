#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
    # cd $GPTUNEROOT/examples/ButterflyPACK
    # rm -rf gptune.db/*.json # do not load any database 
    # tp=ButterflyPACK-IE2D
    # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    # $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./butterflypack_ie2d.py -ntask 1 -nrun 20 -machine tr4 -nprocmin_pernode 2 -optimization GPTune 

    cd $GPTUNEROOT/examples/ButterflyPACK
    rm -rf gptune.db/*.json # do not load any database 
    tp=ButterflyPACK_RFcavity
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    ntask=1
    nrun=10
    nthreads=8

    tuner=GPTune
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./butterflypack_RFcavity_multimode.py -nthreads ${nthreads} -ntask ${ntask} -nrun ${nrun} -optimization $tuner | tee a.out_butterflypackRFcavity_nodes${nodes}_cores${cores}_ntask${ntask}_nrun${nrun}_${tuner}
    # tuner=SIMPLEX
    # $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./butterflypack_RFcavity_multimode_simplex.py -nthreads ${nthreads} -ntask ${ntask} -nrun ${nrun} -optimization $tuner | tee a.out_butterflypackRFcavity_nodes${nodes}_cores${cores}_ntask${ntask}_nrun${nrun}_${tuner}


  # cd $GPTUNEROOT/examples/ButterflyPACK
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=ButterflyPACK_RFcavity_HO
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # ntask=1
  # nrun=10
  # nthreads=1
  # order=0
  # tuner=GPTune
  # $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./butterflypack_RFcavity_multimode_HO.py -order ${order} -nthreads ${nthreads} -ntask ${ntask} -nrun ${nrun} -optimization $tuner | tee a.out_butterflypackRFcavity_nodes${nodes}_cores${cores}_ntask${ntask}_nrun${nrun}_order${order}_${tuner}
  # # tuner=SIMPLEX
  # # $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./butterflypack_RFcavity_multimode_HO_simplex.py -order ${order} -nthreads ${nthreads} -ntask ${ntask} -nrun ${nrun} -optimization $tuner | tee a.out_butterflypackRFcavity_nodes${nodes}_cores${cores}_ntask${ntask}_nrun${nrun}_order${order}_${tuner}




else
    echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi       
fi

