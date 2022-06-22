#!/bin/bash
cd ../../
. run_env.sh
cd -

if [[ -z "${GPTUNE_LITE_MODE}" ]] && [[ $ModuleEnv == *"openmpi"* ]]; then
RUN=$MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1
fi


cd $GPTUNEROOT/examples/GPTune-Demo
tp=GPTune-Demo
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json


##########################################################################################
################## Illustrate basic functionalities ######################################
tuner=GPTune
rm -rf gptune.db/*.json # do not load any database 
$RUN  python ./demo.py -optimization ${tuner} -ntask 2 -nrun 20
###########################################################################################


# the following examples only work with openmpi
if [[ $ModuleEnv == *"openmpi"* ]]; then
    # ###########################################################################################
    # ################### Illustrate parallel modeling and search performance ###################
     # this example performs one MLA iteration using $\epsilon=80$ and $\delta=20$ samples of the analytical function (see Table 2 of the PPoPP paper). Suppose your machine has 1 node with 16 cores, run the following two configurations and compare 'time_search':xxx and 'time_model':xxx from the runlogs.    
     cd $GPTUNEROOT/examples/GPTune-Demo
     rm -rf gptune.db/*.json
     tp=GPTune-Demo
     app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
     echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json  
     $RUN python ./demo_parallelperformance.py -ntask 20 -nrun 40  | tee a.out_seqential # this is the sequential benchmark
     rm -rf gptune.db/*.json
     $RUN python ./demo_parallelperformance.py -ntask 20 -nrun 40 -distparallel 1 | tee a.out_parallel # this is parallel modeling and search

    # ###########################################################################################
fi


# # ###########################################################################################
# # ################### Illustrate use of coarse performance model ############################
# # this example autotunes the analytical function using $\epsilon=20$ and $\delta=2$ with and without a performance model. Suppose your machine has 1 node with 4 cores, run the following two configuratoins and check the difference in "Oopt" and in the plots. You will notice a better optimum is found by using the performance model.  
# cd $GPTUNEROOT/examples/GPTune-Demo
# rm -rf gptune.db/*.json
# tp=GPTune-Demo
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json      
# $RUN python ./demo_perf_model.py -nrun 20 -ntask 2 -perfmodel 0 -plot 1 | tee a.out_demo_perf0 # without a performance model, you will see "Popt  [0.5386916457874029] Oopt  0.5365493976919858" for the task "t:0.000000" and "Popt  [0.466133977547591] Oopt  0.7810616750180267" for the task "t:0.500000"
# rm -rf gptune.db/*.json
# $RUN python ./demo_perf_model.py -nrun 20 -ntask 2 -perfmodel 1 -plot 1 | tee a.out_demo_perf1 # with a performance model, you will see "Popt  [0.5382320616588907] Oopt  0.536732793802327" for the task "t:0.000000" and "Popt  [0.5257108563434262] Oopt  0.5698278466330067" for the task "t:0.500000"
# ###########################################################################################


# ###########################################################################################
# ################### Illustrate multi-fidelity tuning feature of GPTuneBand ################
# Nloop=2
# ntask=1
# plot=0
# restart=1
# expid='0'
# # for tuner in GPTune GPTuneBand hpbandster TPE
# for tuner in GPTuneBand
# do
# rm -rf gptune.db/*.json # do not load any database 
# $RUN  python demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}
# done
# ###########################################################################################





