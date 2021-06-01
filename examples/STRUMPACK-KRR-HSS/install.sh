#!/bin/bash

machine=cori
proc=haswell   # knl,haswell,gpu
mpi=openmpi  # openmpi,craympich
compiler=gnu   # gnu, intel

export ModuleEnv=$machine-$proc-$mpi-$compiler

if [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    PREFIX_PATH=~/.local/cori/3.7-anaconda-2019.10/

    echo $(which python)

    module unload cmake
    module load cmake/3.14.4

    module load gcc/8.3.0
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module load openmpi/4.0.1
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp

    GPTUNEROOT=$PWD/../../
    
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    BLAS_INC="-I${MKLROOT}/include"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
    LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"

    SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"

    MPICC=mpicc
    MPICXX=mpicxx
    MPIF90=mpif90
    OPENMPFLAG=fopenmp

    export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
    export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install
    export METIS_DIR=$ParMETIS_DIR
    export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
    export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
    export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include;$ParMETIS_DIR/include"
    export METIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include"
    export PARMETIS_LIBRARIES=$ParMETIS_DIR/lib/libparmetis.so
    export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.so
fi

rm -rf STRUMPACK
git clone https://github.com/pghysels/STRUMPACK.git
cd STRUMPACK
#git checkout 959ff1115438e7fcd96b029310ed1a23375a5bf6  # head commit has compiler error, requiring fixes
#cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/.
cp ../STRUMPACK-driver/src/KernelRegressionMPI.py examples/.
cp ../STRUMPACK-driver/src/Kernel.cpp src/kernel/.
cp ../STRUMPACK-driver/src/Kernel.h src/kernel/.
chmod +x examples/KernelRegressionMPI.py
mkdir build
cd build

cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_INSTALL_LIBDIR=../install/lib \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DSTRUMPACK_COUNT_FLOPS=ON \
    -DSTRUMPACK_TASK_TIMERS=ON \
    -DSTRUMPACK_USE_CUDA=${STRUMPACK_USE_CUDA} \
    -DTPL_CUBLAS_LIBRARIES="${CUBLAS_LIB}" \
    -DTPL_CUBLAS_INCLUDE_DIRS="${CUBLAS_INCLUDE}" \
    -DCMAKE_CUDA_FLAGS="${STRUMPACK_CUDA_FLAGS}" \
    -DTPL_ENABLE_SCOTCH=ON \
    -DTPL_ENABLE_ZFP=OFF \
    -DTPL_ENABLE_PTSCOTCH=ON \
    -DTPL_ENABLE_PARMETIS=ON \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
    -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
    -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

make install -j32
make examples -j32

cp ../../STRUMPACK-driver/src/STRUMPACKKernel.py ../install/include/python/.
