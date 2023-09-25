#!/bin/bash
cd ../../
. run_env.sh
cd -




# the following examples only work with openmpi and craympich
if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
    rm -rf gptune.db/*.json # do not load any database 
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $RUN  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nprocmin_pernode 1 -ntask 2 -nrun 40 -jobid 0 -tla_I 0 -tla_II 0
    $RUN  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nprocmin_pernode 1 -ntask 2 -nrun 20 -jobid 0 -tla_I 1 -tla_II 0
    $RUN  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nprocmin_pernode 1 -ntask 2 -nrun 20 -jobid 0 -tla_I 0 -tla_II 1
else
    echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi


## use the following command if you want to try sensitivity analysis.
# python scalapack_sensitivity_analysis.py

