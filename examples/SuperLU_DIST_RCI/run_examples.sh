#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"gpu"* ]]; then
  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST_GPU2D
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # bash superlu_MLA_ngpu_RCI.sh -a 10 -b $gpus -c time | tee log.superlu #a: nrun b: nprocmin_pernode c: objective
  # cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)

  cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  rm -rf gptune.db/*.json # do not load any database 
  tp=SuperLU_DIST_MO_GPU2D
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  bash superlu_MLA_ngpu_MO_RCI.sh -a 10 -b $gpus | tee log.superlu_mo_ngpu #a: nrun b: nprocmin_pernode 
  cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)

else
  cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  rm -rf gptune.db/*.json # do not load any database 
  tp=SuperLU_DIST
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  bash superlu_MLA_RCI.sh -a 10 -b 2 -c time | tee log.superlu #a: nrun b: nprocmin_pernode c: objective
  cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_$(timestamp)

  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # bash superlu_MLA_MO_RCI.sh -a 10 -b 2 | tee log.superlu_MO #a: nrun b: nprocmin_pernode 
  # cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_MO_$(timestamp)
fi


