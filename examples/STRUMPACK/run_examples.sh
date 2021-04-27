#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"openmpi"* ]]; then
    cd $GPTUNEROOT/examples/STRUMPACK
    rm -rf gptune.db/*.json # do not load any database 
    tp=STRUMPACK_Poisson3d
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_Poisson3d.py -ntask 1 -nrun 10 -machine cori 

    cd $GPTUNEROOT/examples/STRUMPACK
    rm -rf gptune.db/*.json # do not load any database
    tp=STRUMPACK_KRR
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json 
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR.py -ntask 1 -nrun 10 -machine cori -npernode $cores 




        ntask=1
        nrun=-1
        bmin=2
        bmax=8
        eta=2
        Nloop=1
        restart=1
        dataset="susy_10Kn"
        seed=881
        expname=KRR_${dataset}_ntask${ntask}_bandit${bmin}-${bmax}-${eta}_Nloop${Nloop}
        cd $GPTUNEROOT/examples/STRUMPACK
        rm -rf gptune.db/*.json # do not load any database
        tp=STRUMPACK_KRR
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json 
        
        for expid in TEST1 TEST2 TEST3
        do  
            seed=$( expr ${seed} + 1 )
            
            tuner='GPTune'
            rm gptune.db/STRUMPACK_KRR.json
            $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  \
            python ./strumpack_MLA_KRR_MB.py -ntask ${ntask} -machine cori -npernode $cores -optimization ${tuner}\
            -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -dataset ${dataset} -seed ${seed} -expid ${expid}\
            # 2>&1 | tee a.out_${expname}_expid${expid}_${tuner}


            tuner='GPTuneBand'
            rm gptune.db/STRUMPACK_KRR.json
            $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  \
            python ./strumpack_MLA_KRR_MB.py -ntask ${ntask} -machine cori -npernode $cores -optimization ${tuner}\
            -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -dataset ${dataset} -seed ${seed} -expid ${expid}\
            # 2>&1 | tee a.out_${expname}_expid${expid}_${tuner}
            mpirun -n 1 python strumpack_parse_GPTuneBand_db.py -ntask ${ntask} -save_path ${expname}_expid${expid}_${tuner}


            tuner='hpbandster'
            $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  \
            python ./strumpack_MLA_KRR_MB.py -ntask ${ntask} -machine cori -npernode $cores -optimization ${tuner}\
            -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -dataset ${dataset} -seed ${seed} -expid ${expid}\
            # 2>&1 | tee a.out_${expname}_expid${expid}_${tuner}

            tuner='TPE'
            $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  \
            python ./strumpack_MLA_KRR_MB.py -ntask ${ntask} -machine cori -npernode $cores -optimization ${tuner}\
            -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -dataset ${dataset} -seed ${seed} -expid ${expid}\
            # 2>&1 | tee a.out_${expname}_expid${expid}_${tuner}

            tuner='opentuner'
            $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  \
            python ./strumpack_MLA_KRR_MB.py -ntask ${ntask} -machine cori -npernode $cores -optimization ${tuner}\
            -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -dataset ${dataset} -seed ${seed} -expid ${expid}\
            # 2>&1 | tee a.out_${expname}_expid${expid}_${tuner}

        done


    if [[ $ModuleEnv == *"gpu"* ]]; then
        cd $GPTUNEROOT/examples/STRUMPACK
        rm -rf gptune.db/*.json # do not load any database 
        tp=STRUMPACK_MMdoubleMPIDist_GPU
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_1gpu.py -npernode 1 -ntask 1 -nrun 20
    fi    
fi
