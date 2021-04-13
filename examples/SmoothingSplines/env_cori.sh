#/bin/bash

module load R/3.6.1-conda
module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/../../autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/../../autotune/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/../../scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/../../mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/../../GPTune/
export PYTHONWARNINGS=ignore
