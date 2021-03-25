#!/bin/bash
##SBATCH -C haswell
##SBATCH -J test_driver
##SBATCH --qos=regular
##SBATCH -t 02:00:00
##SBATCH --nodes=2
##SBATCH --mail-user=xz584@cornell.edu
##SBATCH --mail-type=ALL
###SBATCH --ntasks=8
###SBATCH --tasks-per-node=8
##SBATCH --cpus-per-task=8
###SBATCH --constraint=haswell
#
##OpenMP settings:
## export OMP_NUM_THREADS=4
## export OMP_PLACES=threads
## export OMP_PROC_BIND=spread
#
## comment
## 8 MPI processes with 4 OpenMP threads per process, each OpenMP thread has 1 physical core

cd ../../

module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64:/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64

module load openmpi/4.0.1

export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/hypre-driver/
export PYTHONWARNINGS=ignore

CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

nxmax=400
nymax=400
nzmax=400
ntask=1
nrun=10
nprocmin_pernode=32  # nprocmin_pernode=cores means flat MPI 


# test hypredriver, the following calling sequence will first dump the data to file when using GPTune, then read data when using opentuner or hpbandster to make sure they use the same tasks as GPTune
cd -
tuner='GPTune'
cp .gptune/configs/cori.json .gptune/meta.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_ML_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_ntask${ntask}_nrun${nrun}_${tuner}

