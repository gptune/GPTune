#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}




if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
  cd $GPTUNEROOT/examples/MFEM
  rm -rf gptune.db/*.json # do not load any database 
  tp=MFEM
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  $RUN  python ./mfem_maxwell3d.py -ntask 1 -nrun 20 -nprocmin_pernode 2 -optimization GPTune
else
  echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi    

