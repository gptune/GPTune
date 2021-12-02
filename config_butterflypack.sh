#!/bin/bash


export ROOT=$PWD

###################################################
# change these to your compiler and openmpi settings
export CC=gcc-8
export FTN=gfortran-8
export CPP=g++-8
export PATH=$PATH:$ROOT/openmpi-4.1.0/bin
export MPICC="$ROOT/openmpi-4.1.0/bin/mpicc"
export MPICXX="$ROOT/openmpi-4.1.0/bin/mpicxx"
export MPIF90="$ROOT/openmpi-4.1.0/bin/mpif90"
export LD_LIBRARY_PATH=$ROOT/openmpi-4.1.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROOT/openmpi-4.1.0/lib:$LIBRARY_PATH  	
###################################################



export SCALAPACK_LIB=$ROOT/scalapack-2.1.0/build/install/lib/libscalapack.a
export BLAS_LIB=$ROOT/OpenBLAS/libopenblas.a
export LAPACK_LIB=$ROOT/OpenBLAS/libopenblas.a
export LD_LIBRARY_PATH=$ROOT/OpenBLAS/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
OPENMPFLAG=fopenmp


###################################
cd $ROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN install -j32


cd $ROOT
rm -rf scalapack-2.1.0.tgz*
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG" \
	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
make -j32
make install



cd $ROOT
rm -rf ButterflyPACK
git clone https://github.com/liuyangzhuan/ButterflyPACK.git
cd ButterflyPACK
mkdir build
cd build
cmake .. \
	-DCMAKE_Fortran_FLAGS=""\
	-DCMAKE_CXX_FLAGS="" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_INSTALL_LIBDIR=./lib \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
make install -j32

