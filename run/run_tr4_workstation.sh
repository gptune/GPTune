#!/bin/bash

module load gcc/9.1.0
module load openmpi/gcc-9.1.0/4.0.1
module load scalapack-netlib/gcc-9.1.0/2.0.2
module load python/gcc-9.1.0/3.7.4

#shopt -s expand_aliases
#alias python='python3.7'
#alias pip='pip3.7'


export PATH=$PATH:/home/administrator/.local/bin/
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/hypre-driver/
export PYTHONWARNINGS=ignore

CCC=$MPICC
CCCPP=$MPICXX
FTN=$MPIF90
RUN=$MPIRUN

cd examples
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo.py

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_perf_model.py -nrun 20 -machine tr4 -nodes 1 -cores 16 -ntask 5 -perfmodel 0 -plot 0 -optimization GPTune 

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_perf_model.py -nrun 20 -machine tr4 -nodes 1 -cores 16 -ntask 5 -perfmodel 1 -plot 0 -optimization GPTune 

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_comparetuners.py -machine tr4 -nodes 1 -cores 16 -ntask 1 -perfmodel 0 -plot 0 -nrep 2

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine tr4 -jobid 0
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_TLA.py     -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine tr4
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./hypre.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 6 -nxmax 40 -nymax 40 -nzmax 40 -machine tr4
