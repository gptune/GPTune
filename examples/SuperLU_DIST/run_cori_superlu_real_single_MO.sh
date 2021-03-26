#!/bin/bash -l
##SBATCH -q premium
##SBATCH -N 16
##SBATCH -t 2:00:00
##SBATCH -J GPTune_superlu_nimrod
##SBATCH --mail-user=liuyangzhuan@lbl.gov
##SBATCH -C haswell
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

# nodes=16  # number of nodes to be used
# machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
# loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")
# tp=SuperLU_DIST
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json


nprocmin_pernode=1
nrun=20

tuner='GPTune'
cp .gptune/configs/cori.json .gptune/meta.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_single_MO.py -nprocmin_pernode ${nprocmin_pernode} -ntask 1 -nrun ${nrun} -optimization ${tuner} | tee a.out_superlu_single_MO_nprocmin_pernode${nprocmin_pernode}_nrun${nrun}_${tuner}
