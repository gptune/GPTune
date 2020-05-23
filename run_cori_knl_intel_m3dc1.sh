#!/bin/bash
#SBATCH -J Run05D_GPTune
#SBATCH -C knl,quad,cache
#SBATCH -N 128
#SBATCH -q premium
#SBATCH -t 4:00:00
#SBATCH -A m2957
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=4

# --cpus-per-task is 4 x threads-per-mpi on knl

module unload darshan
module swap craype-haswell craype-mic-knl
module load craype-hugepages2M
module unload cray-libsci
module load python/3.7-anaconda-2019.10
module unload cray-mpich
module swap intel intel/19.0.3.199 
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
module load openmpi/4.0.1

export OMPI_MCA_btl_ugni_virtual_device_count=1
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/

export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
# export PYTHONPATH=$PYTHONPATH:$PWD/cython/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore


PETSC_DIR=/project/projectdirs/m2957/liuyangz/my_software/petsc-3-12_superlu-6-2-openmpi-intel-knl
PETSC_ARCH=cori-knl-openmpi401-intel-real-620
HDF5_DIR=${PETSC_DIR}/${PETSC_ARCH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/projectdirs/m2957/liuyangz/my_software/hdf5-1.10.5/usr/local/HDF_Group/HDF5/1.10.5/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HDF5_DIR}/lib
module load gsl
module load idl




CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

cd examples

mpirun -N 32 --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./m3dc1_single_Run05_D_XY_nblock.py -nodes 1792 -cores 2 -ntask 1 -nrun 20 -machine cori -jobid 0 | tee a.out

# mpirun -N 150 --map-by ppr:64:node --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./m3dc1_single_Run05_D_XY_nblock.py -nodes 1792 -cores 2 -ntask 1 -nrun 20 -machine cori -jobid 0
# mpirun -N 15 --map-by ppr:64:node --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./m3dc1_single.py -nodes 224 -cores 2 -ntask 1 -nrun 20 -machine cori -jobid 0
