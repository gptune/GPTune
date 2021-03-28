#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 17
#SBATCH -t 10:00:00
#SBATCH -J GPTune_scalapack
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell
#SBATCH -A m3142

cd ../../

ModuleEnv='cori-haswell-openmpi-gnu'

############### Cori Haswell Openmpi+GNU
if [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-intel PrgEnv-gnu
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    proc=haswell
    cores=32
    machine=cori
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############


export GPTUNEROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONWARNINGS=ignore

cd -

nodes=16  # number of nodes to be used
machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")
tp=PDGEQRF
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json


for nrun in 10
do


mmax=4000
nmax=4000
ntask=2
nrun1=`expr $nrun / 2`
nprocmin_pernode=16  # nprocmin_pernode=cores means flat MPI 


rm -rf gptune.db/PDGEQRF.json
tuner='GPTune'

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax}  -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun1} -nrun1 ${nrun1}  -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_initial
cp gptune.db/PDGEQRF.json . # this will avoid the database file to be polluted

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax}  -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -nrun1 ${nrun1} -perfmodel 1  -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_modelfit

mv PDGEQRF.json gptune.db/.
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax}  -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -nrun1 ${nrun1} -perfmodel 0  -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_nomodel



done
