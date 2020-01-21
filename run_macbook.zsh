#!/bin/zsh

source /usr/local/Cellar/modules/4.3.0/init/zsh

module load gcc/9.2.0
module load openmpi/gcc-9.2.0/4.0.2
module load scalapack/2.1.0
module load python/3.7.6 

#export PATH=$PATH:/home/administrator/.local/bin/
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

CCC=$MPICC
CCCPP=$MPICXX
FTN=$MPIF90
RUN=$MPIRUN

cd examples
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo.py

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine tr4 -jobid 0
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_TLA.py     -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine tr4
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_TLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4
