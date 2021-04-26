#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

cd $GPTUNEROOT/examples/MFEM_RCI
rm -rf gptune.db/MFEM.json # do not load any database 
# cp gptune.db/MFEM.json_memory gptune.db/MFEM.json  
tp=MFEM
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash mfem_maxwell3d_RCI.sh -a 40 -b 2 -c memory | tee log.mfem_memory  #a: nrun b: nprocmin_pernode c: objective
cp gptune.db/MFEM.json  gptune.db/MFEM.json_memory_$(timestamp)

cd $GPTUNEROOT/examples/MFEM_RCI
rm -rf gptune.db/MFEM.json # do not load any database 
# cp gptune.db/MFEM.json_time gptune.db/MFEM.json 
tp=MFEM
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash mfem_maxwell3d_RCI.sh -a 40 -b 2 -c time | tee log.mfem_time #a: nrun b: nprocmin_pernode c: objective
cp gptune.db/MFEM.json  gptune.db/MFEM.json_time_$(timestamp)

cd $GPTUNEROOT/examples/MFEM_RCI
rm -rf gptune.db/MFEM.json # do not load any database 
# cp gptune.db/MFEM.json_time_memory gptune.db/MFEM.json
tp=MFEM
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash mfem_maxwell3d_MO_RCI.sh -a 40 -b 2 | tee log.mfem_time_memory  #a: nrun b: nprocmin_pernode 
cp gptune.db/MFEM.json  gptune.db/MFEM.json_time_memory_$(timestamp)
