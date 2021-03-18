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
# export PYTHONPATH=$PYTHONPATH:$PWD/examples/Scalapack-PDGEQRF/scalapack-driver/spt/
# export PYTHONPATH=$PYTHONPATH:$PWD/examples/Hypre/hypre-driver/
export PYTHONWARNINGS=ignore

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/
export GPTUNEROOT=$PWD


MPICC=$MPICC
MPICXX=$MPICXX
MPIF90=$MPIF90
MPIRUN=$MPIRUN

cd $GPTUNEROOT/examples/GPTune-Demo
$MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo.py

# cd $GPTUNEROOT/examples/GPTune-Demo
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_perf_model.py -nrun 20 -machine tr4 -nodes 1 -cores 16 -ntask 5 -perfmodel 0 -plot 0 -optimization GPTune 

# cd $GPTUNEROOT/examples/GPTune-Demo
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_perf_model.py -nrun 20 -machine tr4 -nodes 1 -cores 16 -ntask 5 -perfmodel 1 -plot 0 -optimization GPTune 

# cd $GPTUNEROOT/examples/GPTune-Demo
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_comparetuners.py -machine tr4 -nodes 1 -cores 16 -ntask 1 -perfmodel 0 -plot 0 -nrep 2

# cd $GPTUNEROOT/examples/GPTune-Demo
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./demo_parallelperformance.py -nrun 100 -machine tr4 -nodes 1 -cores 16 -ntask 1 -perfmodel 0 -distparallel 1


# cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_TLA.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

# cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_TLA_loaddata.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0

# cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine tr4 -jobid 0
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./scalapack_MLA_loaddata.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 40 -machine tr4 -jobid 0


# cd $GPTUNEROOT/examples/SuperLU_DIST
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_single.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4

# cd $GPTUNEROOT/examples/SuperLU_DIST
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_single_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4

# cd $GPTUNEROOT/examples/SuperLU_DIST
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4

# cd $GPTUNEROOT/examples/SuperLU_DIST
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_MO_complex.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine tr4

# cd $GPTUNEROOT/examples/SuperLU_DIST
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./superlu_MLA_TLA.py     -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine tr4



# cd $GPTUNEROOT/examples/Hypre
# $MPIRUN --allow-run-as-root --use-hwthread-cpus -n 1 python ./hypre.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 6 -nxmax 40 -nymax 40 -nzmax 40 -machine tr4

# cd $GPTUNEROOT/examples/ButterflyPACK
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./butterflypack_ie2d.py     -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -nprocmin_pernode 8 -optimization GPTune 

# cd $GPTUNEROOT/examples/ButterflyPACK
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./butterflypack_RFcavity.py -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -nthreads 1 -optimization GPTune 

# cd $GPTUNEROOT/examples/ButterflyPACK
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./butterflypack_RFcavity_multimode.py -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -nthreads 1 -optimization GPTune 

# cd $GPTUNEROOT/examples/STRUMPACK
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./strumpack_MLA_KRR.py  -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -npernode 16 -optimization GPTune

# cd $GPTUNEROOT/examples/STRUMPACK
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./strumpack_MLA_Poisson3d.py -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -nprocmin_pernode 8 -optimization GPTune


# cd $GPTUNEROOT/examples/MFEM
# $MPIRUN --oversubscribe --allow-run-as-root --use-hwthread-cpus -n 1 python ./mfem_maxwell3d.py -nodes 1 -cores 16 -ntask 1 -nrun 20 -machine tr4 -nprocmin_pernode 8 -optimization GPTune