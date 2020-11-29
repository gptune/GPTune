#!/bin/zsh

source /usr/local/Cellar/modules/4.3.0/init/zsh

module load gcc/9.2.0
module load openmpi/gcc-9.2.0/4.0.1
# module load python/3.7.6 
alias python=/usr/local/Cellar/python@3.7/3.7.9_2/bin/python3
alias pip=/usr/local/Cellar/python@3.7/3.7.9_2/bin/pip3
export PATH=$PATH:/usr/local/Cellar/python@3.7/3.7.9_2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/scalapack-2.1.0/build/install/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$PWD/scalapack-2.1.0/build/install/lib
export LIBRARY_PATH=$LIBRARY_PATH:$PWD/scalapack-2.1.0/build/install/lib

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:~/Library/Python/3.7/lib/python/site-packages/
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.7/site-packages
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

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine mac -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine mac -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1300 -nmax 1300 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine mac -jobid 0
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 1300 -nmax 1300 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine mac -jobid 0

$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_TLA.py  -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine mac
$RUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine mac

