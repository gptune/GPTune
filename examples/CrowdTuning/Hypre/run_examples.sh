#!/bin/bash

cp -r ../../Hypre/hypre .
cp -r ../../Hypre/hypre-driver .

cd ../../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
  cd $GPTUNEROOT/examples/CrowdTuning/Hypre-New-120-DefaultParams
  #rm -rf gptune.db/*.json # do not load any database
  #echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json

  for nbatch in 0 1 2 3 4
  do
      tp=Hypre-Full
      tuner=GPTune
      app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
      $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python3 ./hypre_full.py  -nprocmin_pernode 1 -ntask 1 -nrun 40 -nxmax 120 -nymax 120 -nzmax 120 -optimization ${tuner} -nbatch ${nbatch} | tee a.out_hypre_full_${tuner}_${nbatch}
  done

  for nbatch in 0 1 2 3 4
  do
      tp=Hypre-Reduced
      tuner=GPTune
      app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
      $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python3 ./hypre_reduced.py  -nprocmin_pernode 1 -ntask 1 -nrun 40 -nxmax 120 -nymax 120 -nzmax 120 -optimization ${tuner} -nbatch ${nbatch} | tee a.out_hypre_reduced_${tuner}_${nbatch}
  done

else
    echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi  
fi
