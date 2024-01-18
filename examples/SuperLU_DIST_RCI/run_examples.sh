#!/bin/bash
#SBATCH -A m2957
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH --mail-type=BEGIN
#SBATCH -e ./tmp.err

cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [[ $ModuleEnv == *"gpu"* ]]; then
  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST_GPU2D
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # logfile=log.superlu
  # bash superlu_MLA_ngpu_RCI.sh -a 10 -b $gpus -c time | tee ${logfile}  #a: nrun b: nprocmin_pernode c: objective
  # cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)

  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST_MO_GPU2D
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # logfile=log.superlu_mo_ngpu
  # bash superlu_MLA_ngpu_MO_RCI.sh -a 10 -b $gpus | tee ${logfile} #a: nrun b: nprocmin_pernode 
  # cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)

  cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  rm -rf gptune.db/*.json # do not load any database 
  tp=SuperLU_DIST_MO_GPU3D
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  logfile=log.superlu_mo_3d_ngpu
  bash superlu_MLA_3D_ngpu_MO_RCI.sh -a 10 -b $gpus | tee tee ${logfile} #a: nrun b: nprocmin_pernode 
  cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)  

else
  cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  rm -rf gptune.db/*.json # do not load any database 
  tp=SuperLU_DIST
  app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  logfile=log.superlu
  bash superlu_MLA_RCI.sh -a 20 -b 2 -c time | tee ${logfile} #a: nrun b: nprocmin_pernode c: objective
  cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_$(timestamp)

  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # logfile=log.superlu_MO
  # bash superlu_MLA_MO_RCI.sh -a 10 -b 2 | tee ${logfile} #a: nrun b: nprocmin_pernode 
  # cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_MO_$(timestamp)

  # cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
  # rm -rf gptune.db/*.json # do not load any database 
  # tp=SuperLU_DIST_MO_3D
  # app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
  # echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
  # logfile=log.superlu_mo_3d
  # bash superlu_MLA_3D_MO_RCI.sh -a 10 -b $cores | tee ${logfile} #a: nrun b: nprocmin_pernode 
  # cp gptune.db/${tp}.json  gptune.db/${tp}.json_$(timestamp)  


fi


grep time_model log.superlu | tee tmp.out
time_model=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_search log.superlu | tee tmp.out
time_search=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_loaddata log.superlu | tee tmp.out
time_loaddata=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_fun log.superlu | tee tmp.out
time_fun=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

echo "RCI mode stats: time_fun (including srun overhead) ${time_fun}, time_model ${time_model}, time_search ${time_search}, time_loaddata ${time_loaddata}" >> ${logfile}
