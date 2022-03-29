#!/bin/bash -l

PY_VERSION=3.8
PY_TIME=2020.11
MKL_TIME=2020.2.254

module load gcc/8.3.0
module unload cray-mpich
module unload openmpi
module unload PrgEnv-intel
module load PrgEnv-gnu
module load openmpi/4.1.2
module unload craype-hugepages2M
module unload cray-libsci
module unload atp    
module load python/$PY_VERSION-anaconda-$PY_TIME
MPIRUN=mpirun
nodes=4
cores=32
gpus=0

export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/superlu_dist/parmetis-4.0.3/install/lib/

export PYTHONPATH=$PYTHONPATH:../../../autotune/
export PYTHONPATH=$PYTHONPATH:../../../scikit-optimize/
export PYTHONPATH=$PYTHONPATH:../../../mpi4py/
export PYTHONPATH=$PYTHONPATH:../../../GPTune/
export PYTHONPATH=$PYTHONPATH:../../../GPy/
export PYTHONPATH=$PYTHONPATH:../../../pygmo2/
export PYTHONWARNINGS=ignore
export GPTUNEROOT=$PWD

#for nbatch in {0..9}
#for matname in Si2.mtx SiH4.mtx SiNa.mtx benzene.mtx Na5.mtx Si5H12.mtx Si10H16.mtx SiO.mtx H2O.mtx GaAsH6.mtx Ga3As3H12.mtx
#for matname in Si5H12.mtx Si10H16.mtx SiO.mtx H2O.mtx GaAsH6.mtx Ga3As3H12.mtx
for matname in SiO.mtx H2O.mtx Si5H12.mtx Si10H16.mtx GaAsH6.mtx Ga3As3H12.mtx
do
    for nbatch in 0 1 2
    do
        for experiment in "tuning_all" "tuning_reduced_default" "tuning_reduced_random"
        do
            echo "Run experiment=${experiment} matname=${matname} nbatch=${nbatch}"
            mpirun --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_tuning.py -experiment ${experiment} -matname ${matname} -nbatch ${nbatch} -nrun 20 -npilot 0
        done
    done
done
echo "\n"

