#!/bin/zsh

export GPTUNEROOT=$PWD
export PATH=$GPTUNEROOT/env/bin/:$PATH
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPTUNEROOT/openmpi-4.0.1/lib
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/pygmo2/build/
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$GPTUNEROOT/scalapack-2.1.0/build/install/lib/
export PYTHONWARNINGS=ignore

export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export MPIRUN="$GPTUNEROOT/openmpi-4.0.1/bin/mpirun"

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


