#!/bin/bash
cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

cd $GPTUNEROOT/examples/Scalapack-PDGEQRF_RCI
rm -rf gptune.db/*.json # do not load any database 
tp=PDGEQRF
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash scalapack_MLA_RCI.sh -a 40 -b 2 -c 1000 -d 1000 -e 2 | tee log.pdgeqrf #a: nrun b: nprocmin_pernode c: mmax d: nmax e: ntask
cp gptune.db/PDGEQRF.json  gptune.db/PDGEQRF.json_$(timestamp)
