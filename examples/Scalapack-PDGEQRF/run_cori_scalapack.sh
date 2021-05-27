#!/bin/bash -l

cd ../../

if [[ ${1} != "" ]]; then
    Scenario=${1}
else
    Scenario='MLA'
fi

if [[ ${2} != "" ]]; then
    Config=${2}
else
    Config='cori-haswell-openmpi-gnu'
fi

if [[ $Config == *"cori-haswell-openmpi-gnu"* ]]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-intel PrgEnv-gnu
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
elif [[ $Config = *"cori-knl-openmpi-gnu"* ]]; then
    module load python/3.7-anaconda-2019.10
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-intel PrgEnv-gnu
	module load openmpi/4.0.1
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
else
    echo "Untested Config: $Config, please add the corresponding definitions in this file"
    exit
fi

export GPTUNEROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONWARNINGS=ignore

cd -

if [[ ${3} != "" ]]; then
    mmax=${3}
    nmax=${3}
else
    mmax=10000
    nmax=10000
fi

if [[ ${4} != "" ]]; then
    ntask=${4}
else
    ntask=1
fi

if [[ ${5} != "" ]]; then
    nrun=${5}
else
    nrun=20
fi

if [[ ${6} != "" ]]; then
    nprocmin_pernode=${6}
else
    nprocmin_pernode=1
fi

if [[ ${7} != "" ]]; then
    optimization=${7}
else
    optimization='GPTune'
fi

if [[ ${Scenario} == "MLA" ]]; then
    cp .gptune/configs/${Config}.json .gptune/meta.json

    mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python3 ./scalapack_${Scenario}.py -nprocmin_pernode ${nprocmin_pernode} -optimization ${optimization} -mmax ${mmax} -nmax ${nmax} -ntask ${ntask} -nrun ${nrun} | tee a.out.log

elif [[ ${Scenario} == "TLA_task" ]]; then
    if [[ ${8} != "" ]]; then
        tvalue=${8}
    else
        tvalue=0
    fi
    if [[ ${9} != "" ]]; then
        tvalue2=${9}
    else
        tvalue2=0
    fi
    if [[ ${10} != "" ]]; then
        tvalue3=${10}
    else
        tvalue3=0
    fi
    if [[ ${11} != "" ]]; then
        tvalue4=${11}
    else
        tvalue4=0
    fi
    if [[ ${12} != "" ]]; then
        tvalue5=${12}
    else
        tvalue5=0
    fi

    cp .gptune/configs/${Config}.json .gptune/meta.json

    mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python3 ./scalapack_${Scenario}.py -nprocmin_pernode ${nprocmin_pernode} -optimization ${optimization} -mmax ${mmax} -nmax ${nmax} -ntask ${ntask} -nrun ${nrun} -tvalue ${tvalue} -tvalue2 ${tvalue2} -tvalue3 ${tvalue3} -tvalue4 ${tvalue4} -tvalue5 ${tvalue5} | tee a.out.log

elif [[ ${Scenario} == "TLA_machine" ]]; then

    cp .gptune/configs/${Config}.json .gptune/meta.json

    mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python3 ./scalapack_${Scenario}.py -nprocmin_pernode ${nprocmin_pernode} -optimization ${optimization} -mmax ${mmax} -nmax ${nmax} -ntask ${ntask} -nrun ${nrun} | tee a.out.log
fi

