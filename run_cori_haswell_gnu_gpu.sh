#!/bin/bash
module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load cuda/10.2.89 
# module load openmpi/4.0.3

export LD_LIBRARY_PATH=/usr/common/software/cuda/10.2.89/extras/CUPTI/lib64:/usr/common/software/cuda/10.2.89/lib64:/usr/common/software/cudnn/7.6.5/cuda/10.2.89/lib64:/global/common/cori_cle7/software/jdk/1.8.0_202/lib:/opt/cray/job/2.2.4-7.0.1.1_3.40__g36b56f4.ari/lib64:/opt/esslurm/lib64:/opt/gcc/8.3.0/snos/lib64:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64:/usr/common/software/cuda/10.2.89/lib64/stubs:/usr/common/software/sles15_cgpu/ucx/1.9.0/lib
export PATH=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89:$PATH
mpicc=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpicc
mpicxx=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpicxx
mpif90=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpif90
mpirun=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpirun


export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore

CCC=$mpicc
CCCPP=$mpicxx
FTN=$mpif90

cd examples
# srun -n 1  python ./demo.py
# $mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./demo.py

# $mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0

# $mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0

# $mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 20 -machine cori -jobid 0
# $mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0

# $mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA_TLA.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 20 -machine cori
$mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_single_1gpu.py  -nodes 1 -cores 2 -nprocmin_pernode 1 -ntask 1 -nrun 40 -machine cori
