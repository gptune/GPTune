#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 8
#SBATCH -t 10:00:00
#SBATCH -J GPTune_scalapack
#SBATCH --mail-user=younghyun@berkeley.edu
#SBATCH -C haswell

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

cd examples

#mkdir loadmodel_eval
#
#ntask=5
#nrun=200
#nodes=8
#cores=32
#nprocmin_pernode=1
#
#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe
#cp scalapack-pdqrdriver.json loadmodel_eval/scalapack-pdqrdriver.first.json
#cp modeling_stat_scalapack.csv loadmodel_eval/modeling_stat_scalapack.csv
#
#rm scalapack-pdqrdriver.json
#rm modeling_stat_scalapack.csv
#cp loadmodel_eval/scalapack-pdqrdriver.first.json scalapack-pdqrdriver.json
#
#nrun=50
#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
#cp scalapack-pdqrdriver.json loadmodel_eval/scalapack-pdqrdriver.history_db.json
#cp modeling_stat_scalapack.csv loadmodel_eval/modeling_stat_scalapack.history_db.csv
#
#rm scalapack-pdqrdriver.json
#rm modeling_stat_scalapack.csv
#cp loadmodel_eval/scalapack-pdqrdriver.first.json scalapack-pdqrdriver.json
#
#nrun=50
#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_max_eval
#cp scalapack-pdqrdriver.json loadmodel_eval/scalapack-pdqrdriver.loadmodel.json
#cp modeling_stat_scalapack.csv loadmodel_eval/modeling_stat_scalapack.loadmodel.csv
#
#rm scalapack-pdqrdriver.json
#rm modeling_stat_scalapack.csv
#cp loadmodel_eval/scalapack-pdqrdriver.first.json scalapack-pdqrdriver.json



#mkdir loadmodel_eval2
#
ntask=5
nrun=50
nodes=8
cores=32
nprocmin_pernode=1
#
#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe
#
#cp scalapack-pdqrdriver.json loadmodel_eval2/scalapack-pdqrdriver.first.json
#cp modeling_stat_scalapack.csv loadmodel_eval2/modeling_stat_scalapack.csv
#
#rm scalapack-pdqrdriver.json
#rm modeling_stat_scalapack.csv
#
#cp loadmodel_eval2/scalapack-pdqrdriver.first.json scalapack-pdqrdriver.json
#
#nrun=50
#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
#
#cp scalapack-pdqrdriver.json loadmodel_eval2/scalapack-pdqrdriver.history_db.json
#cp modeling_stat_scalapack.csv loadmodel_eval2/modeling_stat_scalapack.history_db.csv
#
#rm scalapack-pdqrdriver.json
#rm modeling_stat_scalapack.csv

cp loadmodel_eval2/scalapack-pdqrdriver.first.json scalapack-pdqrdriver.json

nrun=50
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_max_eval

cp scalapack-pdqrdriver.json loadmodel_eval2/scalapack-pdqrdriver.loadmodel.json
cp modeling_stat_scalapack.csv loadmodel_eval2/modeling_stat_scalapack.loadmodel.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

