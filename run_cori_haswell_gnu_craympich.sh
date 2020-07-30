#!/bin/bash
export CRAYPE_LINK_TYPE=dynamic
module swap cray-mpich cray-mpich/7.7.12
module load python/3.7-anaconda-2019.07
# module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
# module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore


cd examples
srun -n 1  python ./demo.py

srun -n 1  python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine cori -jobid 0

# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine cori -jobid 0

# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine cori -jobid 0
# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine cori -jobid 0

# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA_TLA.py  -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine cori
