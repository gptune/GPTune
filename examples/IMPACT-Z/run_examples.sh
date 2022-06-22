#!/bin/bash
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH -J GPTune
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell

cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
  cd $GPTUNEROOT/examples/IMPACT-Z
  rm -rf gptune.db/*.json # do not load any database 
  tp=IMPACT-Z
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  
  # tuner=GPTune
  # # # $MPIRUN --oversubscribe --allow-run-as-root --mca btl self,tcp,vader --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./impact-z_single1D.py -ntask 1 -nrun 200 -optimization $tuner | tee a.out1D_${tuner}
  # $MPIRUN --oversubscribe --allow-run-as-root --mca btl self,tcp,vader --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./impact-z_single.py -ntask 1 -nrun 20 -optimization $tuner | tee a.out_${tuner}


  tuner=SIMPLEX
  $MPIRUN --oversubscribe --allow-run-as-root --mca btl self,tcp,vader --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./impact-z_single_simplex.py -ntask 1 -nrun 100 -optimization $tuner | tee a.out_${tuner}
else
    echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi  
fi
