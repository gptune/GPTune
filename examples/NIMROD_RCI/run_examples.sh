#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

cd $GPTUNEROOT/examples/NIMROD_RCI
tp=NIMROD
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json

seed=881
for expid in 1 2 3
do  
    rm -rf gptune.db/*.json # do not load any database 
    seed=$( expr ${seed} + ${expid} )
    nstepmax=30
    nstepmin=3
    Nloop=1
    optimization='GPTune'
    bash nimrod_single_MB_RCI.sh -a $nstepmax -b $nstepmin -c $Nloop -d $optimization -e $expid -f $seed | tee log.nimrod_nstepmax${nstepmax}_nstepmin${nstepmin}_Nloop${Nloop}_optimization${optimization}_nodes${nodes}_expid${expid}_seed${seed} #a: nstepmax b: nstepmin c: Nloop d: optimization e: expid f: seed
    cp gptune.db/NIMROD.json  gptune.db/NIMROD.json_$(timestamp)
done
