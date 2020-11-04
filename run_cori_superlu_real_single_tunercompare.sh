#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 64
#SBATCH -t 3:00:00
#SBATCH -J GPTune_superlu
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell
#SBATCH -A m2957

module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64


module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONWARNINGS=ignore

CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

cd examples
nodes=63
cores=32
nprocmin_pernode=4
nrun=20
ntask=8

rm -rf *.pkl
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_single.py  -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori -optimization ${tuner} | tee a.out_superlu_MLA_SO_nodes${nodes}_cores${cores}_nprocmin_pernode${nprocmin_pernode}_nrun${nrun}_${tuner}
tuner='opentuner'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_single.py  -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori -optimization ${tuner} | tee a.out_superlu_MLA_SO_nodes${nodes}_cores${cores}_nprocmin_pernode${nprocmin_pernode}_nrun${nrun}_${tuner}
tuner='hpbandster'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_single.py  -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori -optimization ${tuner} | tee a.out_superlu_MLA_SO_nodes${nodes}_cores${cores}_nprocmin_pernode${nprocmin_pernode}_nrun${nrun}_${tuner}
