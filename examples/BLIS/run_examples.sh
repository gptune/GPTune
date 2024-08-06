#!/bin/bash
cd ../../
. run_env.sh
cd -


cd $GPTUNEROOT/examples/BLIS
tp=BLISv2
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json


##########################################################################################
################## Comparing GPTune with cGP ######################################
tuner=GPTune
rm -rf gptune.db/*.json # do not load any database
python ./blisv2.py -optimization ${tuner} -nrun 40 | tee log.blisv2_${tuner}

tuner=cgp
rm -rf gptune.db/*.json # do not load any database
python ./blisv2.py -optimization ${tuner} -nrun 40 | tee log.blisv2_${tuner}
###########################################################################################
