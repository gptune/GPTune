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

mkdir scalapack_loadmodel_eval

ntask=5
firstrun=100
nodes=8
cores=32
nprocmin_pernode=1

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${firstrun} -machine cori | tee scalapack_loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${firstrun}_oversubscribe
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval/scalapack-pdqrdriver.first_nrun${firstrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval/modeling_stat_scalapack_nrun${firstrun}csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval/scalapack-pdqrdriver.history_db_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval/modeling_stat_scalapack.history_db_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval/scalapack-pdqrdriver.loadmodel_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval/modeling_stat_scalapack.loadmodel_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

################
mkdir scalapack_loadmodel_eval2

ntask=5
firstrun=50
nodes=8
cores=32
nprocmin_pernode=1

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${firstrun} -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${firstrun}_oversubscribe
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack_nrun${firstrun}csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.history_db_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack.history_db_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.loadmodel_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack.loadmodel_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

firstrun=50
nrun=25
cp scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.history_db_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack.history_db_nrun${nrun}.csv

firstrun=50
nrun=25
cp scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -update -1 -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_update_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.loadmodel_nrun${nrun}_update.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack.loadmodel_nrun${nrun}_update.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

firstrun=50
nrun=25
cp scalapack_loadmodel_eval2/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -update 5 -machine cori | tee scalapack_loadmodel_eval2/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_update5_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval2/scalapack-pdqrdriver.loadmodel_nrun${nrun}_update5.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval2/modeling_stat_scalapack.loadmodel_nrun${nrun}_update5.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

#################
mkdir scalapack_loadmodel_eval3

ntask=5
firstrun=20
nodes=8
cores=32
nprocmin_pernode=1

mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${firstrun} -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${firstrun}_oversubscribe
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack_nrun${firstrun}csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.history_db_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack.history_db_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

nrun=1
cp scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.loadmodel_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack.loadmodel_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

firstrun=20
nrun=10
cp scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_history_db.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_history_db
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.history_db_nrun${nrun}.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack.history_db_nrun${nrun}.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

firstrun=20
nrun=10
cp scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -update -1 -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_update_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.loadmodel_nrun${nrun}_update.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack.loadmodel_nrun${nrun}_update.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

firstrun=20
nrun=10
cp scalapack_loadmodel_eval3/scalapack-pdqrdriver.first_nrun${firstrun}.json scalapack-pdqrdriver.json
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA_loadmodel.py -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -update 5 -machine cori | tee scalapack_loadmodel_eval3/a.out_scalapck_MLA_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_oversubscribe_loadmodel_update5_max_eval
cp scalapack-pdqrdriver.json scalapack_loadmodel_eval3/scalapack-pdqrdriver.loadmodel_nrun${nrun}_update5.json
cp modeling_stat_scalapack.csv scalapack_loadmodel_eval3/modeling_stat_scalapack.loadmodel_nrun${nrun}_update5.csv

rm scalapack-pdqrdriver.json
rm modeling_stat_scalapack.csv

