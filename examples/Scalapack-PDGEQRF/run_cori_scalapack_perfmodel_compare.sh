#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 17
#SBATCH -t 10:00:00
#SBATCH -J GPTune_scalapack
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell
#SBATCH -A m3142

cd ../../
module load python/3.7-anaconda-2019.10
module unload cray-mpich/7.7.6

module swap PrgEnv-intel PrgEnv-gnu
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

cd -

for nrun in 10
do


mmax=4000
nmax=4000
ntask=1
nruns1=`expr $nrun / 2`
nodes=16
cores=32
nprocmin_pernode=16  # nprocmin_pernode=cores means flat MPI 


rm -rf gptune.db/PDGEQRF.json
tuner='GPTune'

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nruns1} -nruns1 ${nruns1} -machine cori1 -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_initial
cp gptune.db/PDGEQRF.json . # this will avoid the database file to be polluted

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -nruns1 ${nruns1} -perfmodel 1 -machine cori1 -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_modelfit

mv PDGEQRF.json gptune.db/.
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA_perfmodel.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -nruns1 ${nruns1} -perfmodel 0 -machine cori1 -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_nomodel



done
