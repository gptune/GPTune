#!/bin/bash
#SBATCH -J Run05B_GPTune
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -A m2957


module unload darshan
module swap craype-haswell craype-mic-knl
module load craype-hugepages2M
module unload cray-libsci

module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
# export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
module load openmpi/4.0.1
export OMPI_MCA_btl_ugni_virtual_device_count=1

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore


PETSC_DIR=/project/projectdirs/m2957/liuyangz/my_software/petsc-3-12_superlu-6-2-openmpi-knl
PETSC_ARCH=cori-knl-openmpi401-real-620
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

mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./m3dc1_single.py -nodes 1 -cores 17 -ntask 1 -nrun 20 -machine cori -jobid 0
