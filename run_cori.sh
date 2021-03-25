#!/bin/bash


# ModuleEnv='yang-tr4-openmpi-gnu'
ModuleEnv='cori-haswell-openmpi-gnu'
# ModuleEnv='cori-haswell-openmpi-intel'
# ModuleEnv='cori-knl-openmpi-gnu'
# ModuleEnv='cori-knl-openmpi-intel'



############### Yang's tr4 machine
if [ $ModuleEnv = 'yang-tr4-openmpi-gnu' ]; then
    module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load python/gcc-9.1.0/3.7.4
    proc=amd
    cores=16
    machine=tr4
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
fi
###############


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
fi    
###############

############### Cori Haswell Openmpi+Intel
if [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    proc=haswell
    cores=32
    machine=cori
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")    
fi    
###############


############### Cori KNL Openmpi+GNU
if [ $ModuleEnv = 'cori-knl-openmpi-gnu' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-intel PrgEnv-gnu
	module load openmpi/4.0.1
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    proc=knl
    cores=64
    machine=cori
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")    
fi    
###############


############### Cori KNL Openmpi+Intel
if [ $ModuleEnv = 'cori-knl-openmpi-intel' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
	module load openmpi/4.0.1
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    proc=knl
    cores=64
    machine=cori
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")       
fi    
###############

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD



nodes=1  # number of nodes to be used
machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")



# cd $GPTUNEROOT/examples/GPTune-Demo
# rm -rf gptune.db/*.json # do not load any database 
# tp=GPTune-Demo
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./demo.py

# cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
# rm -rf gptune.db/*.json # do not load any database 
# tp=PDGEQRF
# app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
# echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0 -tla 0
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 20 -machine cori -jobid 0 -tla 1

cd $GPTUNEROOT/examples/SuperLU_DIST
rm -rf gptune.db/*.json # do not load any database 
tp=SuperLU_DIST
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 20 -machine cori


# cd $GPTUNEROOT/examples/STRUMPACK
# rm -rf gptune.db/*.json # do not load any database 
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_Poisson3d.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 


###### the one has a segmentation fault when running on Cori
# cd $GPTUNEROOT/examples/STRUMPACK
# rm -rf gptune.db/*.json # do not load any database 
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 


