#!/bin/bash
#SBATCH -A m3142
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --mail-type=BEGIN
#SBATCH -e ./tmp.err

cd ../../
. run_env.sh
cd -

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}


cd $GPTUNEROOT/examples/QTT-FDTD
rm -rf gptune.db/*.json # do not load any database 
tp=QTT-FDTD
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
logfile=log.qtt-fdtd
bash QTT_MLA_MO_RCI.sh -a 20 -b 1 | tee ${logfile} #a: nrun b: nprocmin_pernode 
cp gptune.db/QTT-FDTD.json  gptune.db/QTT-FDTD.json_$(timestamp)

# cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
# rm -rf gptune.db/*.json # do not load any database 
# tp=SuperLU_DIST
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
# logfile=log.superlu_MO
# bash superlu_MLA_MO_RCI.sh -a 10 -b 2 | tee ${logfile} #a: nrun b: nprocmin_pernode 
# cp gptune.db/SuperLU_DIST.json  gptune.db/SuperLU_DIST.json_MO_$(timestamp)


grep time_model log.qtt-fdtd > tmp.out
time_model=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_search log.qtt-fdtd > tmp.out
time_search=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_loaddata log.qtt-fdtd > tmp.out
time_loaddata=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

grep time_fun log.qtt-fdtd > tmp.out
time_fun=$(python -c "import numpy as np;
data=np.loadtxt('tmp.out',dtype='str');
time=np.sum(data[:,1].astype(np.float32))
print(time)")

echo "RCI mode stats: time_fun (including srun overhead) ${time_fun}, time_model ${time_model}, time_search ${time_search}, time_loaddata ${time_loaddata}" >> ${logfile}
