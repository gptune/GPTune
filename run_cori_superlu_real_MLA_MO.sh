#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 16
#SBATCH -t 2:00:00
#SBATCH -J GPTune_superlu_nimrod
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell

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

nodes=8
cores=32
nprocmin_pernode=1
nrun=80

rm -rf *.pkl
tuner='GPTune'
# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_single.py  -nodes 2 -cores 32 -nprocmin_pernode 1 -ntask 1 -nrun 40 -machine cori -optimization ${tuner}
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_MLA_MO.py  -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask 1 -nrun ${nrun} -machine cori -optimization ${tuner} | tee a.out_superlu_MLA_MO_nodes${nodes}_cores${cores}_nprocmin_pernode${nprocmin_pernode}_nrun${nrun}_${tuner}
# mpirun --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true --mca routed radix  -n 1 python ./superlu_MLA_MO_complex.py  -nodes 128 -cores 2 -ntask 1 -nrun 40 -machine cori
