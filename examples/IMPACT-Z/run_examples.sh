#!/bin/bash
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH -J GPTune
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell

cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}




if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
  cd $GPTUNEROOT/examples/IMPACT-Z
  rm -rf gptune.db/*.json # do not load any database 
  tp=IMPACT-Z
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  
  tuner=GPTune
  # # $RUN  python ./impact-z_single1D.py -ntask 1 -nrun 200 -optimization $tuner | tee a.out1D_${tuner}
  $RUN  python ./impact-z_single.py -ntask 1 -nrun 20 -optimization $tuner | tee a.out_${tuner}


  # tuner=SIMPLEX
  # $RUN python ./impact-z_single_simplex.py -ntask 1 -nrun 100 -optimization $tuner | tee a.out_${tuner}
else
    echo "GPTUNE_LITE_MODE cannot run MPI_spawn invoked applications"
fi  

