#!/bin/bash
module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
# export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD

MPICC=mpicc
MPICXX=mpicxx
MPIF90=mpif90

cd $GPTUNEROOT/examples/GPTune-Demo
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./demo.py
cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0
cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0
cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 20 -machine cori -jobid 0
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0
cd $GPTUNEROOT/examples/SuperLU_DIST
mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA_TLA.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 20 -machine cori

# cd $GPTUNEROOT/examples/STRUMPACK
# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_Poisson3d.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 


###### the one has a segmentation fault when running on Cori
# cd $GPTUNEROOT/examples/STRUMPACK
# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 


