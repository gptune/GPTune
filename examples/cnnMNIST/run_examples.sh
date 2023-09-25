#!/bin/bash
cd ../../
. run_env.sh
cd -



############ On Cori, this require pytorch (change run_env.sh and config_cori.sh accordingly)

# the following examples only work with openmpi and craympich
if [[ $ModuleEnv == *"openmpi"* ]] || [[ $ModuleEnv == *"craympich"* ]]; then
    if [[ $ModuleEnv == *"gpu"* ]]; then
        device='cuda'
    else
        device='cpu'
    fi
    cd $GPTUNEROOT/examples/cnnMNIST
    rm -rf gptune.db/*.json # do not load any database
    tp=cnnMNIST
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json 
    $RUN python ./cnnMNIST_MB.py -ntask 1 -nrun -1 -machine cori -npernode $cores -optimization GPTuneBand \
    -bmin 3 -bmax 27 -eta 3 -Nloop 1 -ntrain 8192 -nvalid 1024 -device $device
fi