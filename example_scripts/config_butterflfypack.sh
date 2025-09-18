#!/bin/bash

##################################################
##################################################


export GPTUNEROOT=$PWD

############### Yang's tr4 machine

CC=gcc-8
FTN=gfortran-8
CPP=g++-8

# set the following manually
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  	


export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.so
export BLAS_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
export LAPACK_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
export LD_LIBRARY_PATH=$GPTUNEROOT/OpenBLAS/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
OPENMPFLAG=fopenmp


###############





# install dependencies using apt-get and virtualenv
###################################

apt-get update -y 
apt-get upgrade -y 
apt-get dist-upgrade -y  
apt-get install dialog apt-utils -y 
apt-get install build-essential software-properties-common -y 
add-apt-repository ppa:ubuntu-toolchain-r/test -y 
apt-get update -y 
apt-get install gcc-8 g++-8 gfortran-8 -y  
# apt-get install gcc-9 g++-9 gfortran-9 -y  
# apt-get install gcc-10 g++-10 gfortran-10 -y  


apt-get install libffi-dev -y
apt-get install libssl-dev -y

# apt-get install libblas-dev  -y
# apt-get install liblapack-dev -y
apt-get install cmake -y
apt-get install git -y
apt-get install vim -y
apt-get install autoconf automake libtool -y
apt-get install zlib1g-dev -y
apt-get install wget -y
apt-get install libsm6 -y
apt-get install libbz2-dev -y
apt-get install libsqlite3-dev -y
apt-get install jq -y


cd $GPTUNEROOT
apt purge --auto-remove cmake -y
version=3.19
build=1
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
./bootstrap
make -j32
make install
export PATH=$GPTUNEROOT/cmake-$version.$build/bin/:$PATH




# manually install dependencies from cmake and make
###################################
cd $GPTUNEROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN install -j32



cd $GPTUNEROOT
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
bzip2 -d openmpi-4.0.1.tar.bz2
tar -xvf openmpi-4.0.1.tar 
cd openmpi-4.0.1/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
make -j32
make install



cd $GPTUNEROOT
rm -rf scalapack-2.1.0.tgz*
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=ON \
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




cd $GPTUNEROOT/examples/ButterflyPACK
rm -rf ButterflyPACK
git clone https://github.com/liuyangzhuan/ButterflyPACK.git
cd ButterflyPACK
mkdir build
cd build
cmake .. \
	-DCMAKE_CXX_FLAGS="" \
	-DBUILD_SHARED_LIBS=ON \
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






