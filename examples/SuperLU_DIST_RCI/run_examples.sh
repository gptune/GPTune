#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
rm -rf gptune.db/*.json # do not load any database 
tp=SuperLU_DIST
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash superlu_MLA_RCI.sh -a 10 -b 2 -c memory | tee log.superlu #a: nrun b: nprocmin_pernode c: objective
cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_$(timestamp)

cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
rm -rf gptune.db/*.json # do not load any database 
tp=SuperLU_DIST
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash superlu_MLA_MO_RCI.sh -a 10 -b 2 | tee log.superlu_MO #a: nrun b: nprocmin_pernode 
cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_MO_$(timestamp)