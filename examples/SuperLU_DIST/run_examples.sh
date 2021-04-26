#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then

  cd $GPTUNEROOT/examples/SuperLU_DIST
  rm -rf gptune.db/*.json # do not load any database 
  tp=SuperLU_DIST
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA.py -nprocmin_pernode 1 -ntask 1 -nrun 20 -machine cori


  if [[ $ModuleEnv == *"gpu"* ]]; then
      cd $GPTUNEROOT/examples/SuperLU_DIST
      rm -rf gptune.db/*.json # do not load any database 
      tp=SuperLU_DIST_GPU
      app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
      echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
      $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA_1gpu.py -npernode 1 -ntask 1 -nrun 20 -obj "time"

      cd $GPTUNEROOT/examples/SuperLU_DIST
      rm -rf gptune.db/*.json # do not load any database 
      tp=SuperLU_DIST_GPU
      app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
      echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
      $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA_ngpu.py -npernode 8 -ntask 1 -nrun 20 -obj "time"   
  fi        
fi
