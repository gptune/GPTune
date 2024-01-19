#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}




# cd $GPTUNEROOT/examples/SuperLU_DIST
# rm -rf gptune.db/*.json # do not load any database 
# tp=SuperLU_DIST_MO
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
# $RUN python ./superlu_MLA_MO.py -nprocmin_pernode 1  -ntask 3 -nrun 10 | tee a.out_superlu_multiobj


 cd $GPTUNEROOT/examples/SuperLU_DIST
#  rm -rf gptune.db/*.json # do not load any database 
 tp=SuperLU_DIST
 tuner=GPTune #cgp 
 app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
 echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
 $RUN  python ./superlu_MLA.py -nprocmin_pernode 1 -ntask 1 -nrun 20 -optimization ${tuner} | tee a.out


if [[ $ModuleEnv == *"gpu"* ]]; then
    cd $GPTUNEROOT/examples/SuperLU_DIST
    rm -rf gptune.db/*.json # do not load any database 
    tp=SuperLU_DIST_GPU2D
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $RUN  python ./superlu_MLA_1gpu.py -npernode 1 -ntask 1 -nrun 20 -obj "time"

    cd $GPTUNEROOT/examples/SuperLU_DIST
    rm -rf gptune.db/*.json # do not load any database 
    tp=SuperLU_DIST_GPU2D
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $RUN  python ./superlu_MLA_ngpu.py -npernode 8 -ntask 1 -nrun 20 -obj "time"   
fi        

