#!/bin/bash
cd ../../
. run_env.sh
cd -

# the following examples only work with openmpi
if [[ $ModuleEnv == *"openmpi"* ]]; then
    cd $GPTUNEROOT/examples/NIMROD
    rm -rf gptune.db/*.json # do not load any database 
    tp=NIMROD
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    
    ntask=1
    tuner='GPTune'
    nstepmax=30
    nstepmin=3
    Nloop=1
    bmin=1
    bmax=8
    eta=2   
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./nimrod_single_MB.py -bmin ${bmin} -bmax ${bmax} -eta ${eta} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -nstepmax ${nstepmax} -nstepmin ${nstepmin} | tee a.out_nimrod_single_MB_nstepmax${nstepmax}_nstepmin${nstepmin}_Nloop${Nloop}_tuner${tuner}_bmin${bmin}_bmax${bmax}_eta${eta}
fi