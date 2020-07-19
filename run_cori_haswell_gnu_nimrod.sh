#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 9
#SBATCH -t 10:00:00
#SBATCH -J GPTune_nimrod
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell
#SBATCH -A m2956


module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

module load openmpi/4.0.1

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

ntask=1
nrun=80
nodes=128
cores=2
tuner='GPTune'
mpirun --mca btl self,tcp,vader --oversubscribe -N 32 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./nimrod_single.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 | tee a.out_nimrod_single_ntask${ntask}_nrun${nrun}_nstart8

# ntask=4
# nrun=20
# mpirun --mca btl self,tcp,vader --oversubscribe -N 32 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./nimrod_MLA.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 | tee a.out_nimrod_MLA_ntask${ntask}_nrun${nrun}_nstart8


# ntask=1
# nrun=20
# mpirun --mca btl self,tcp,vader --oversubscribe -N 32 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./nimrod_single.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 | tee a.out_nimrod_single_ntask${ntask}_nrun${nrun}_nstart8


