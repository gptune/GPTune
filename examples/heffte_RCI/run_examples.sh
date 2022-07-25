#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}


  cd $GPTUNEROOT/examples/heffte_RCI
  rm -rf gptune.db/*.json # do not load any database 
  mkdir -p .gptune
  tp=heFFTe
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  bash heffte_RCI.sh -a 20 -b $cores | tee log.heffte #a: nrun b: npernode (set to number of cores per node to do fully packed MPI for now) 
  cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)  




