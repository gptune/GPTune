#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
  cd $GPTUNEROOT/examples/Hypre
  rm -rf gptune.db/*.json # do not load any database 
  tp=Hypre 
  tuner=GPTune
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json                  
  $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py  -nprocmin_pernode 1 -ntask 1 -nrun 10 -nxmax 40 -nymax 40 -nzmax 40 -optimization ${tuner} | tee a.out_hypre_${tuner} 


  cd $GPTUNEROOT/examples/Hypre
  rm -rf gptune.db/*.json # do not load any database 
  tp=Hypre 
  tuner=GPTuneBand
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json                  
  $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py  -nprocmin_pernode 1 -bmin 1 -bmax 8 -eta 2 -amin 0.1 -amax 0.8 -cmin 0.1 -cmax 0.8 -ntask 2 -Nloop 1 -optimization ${tuner} | tee a.out_hypre_MB_${tuner}       
fi
