#!/bin/bash

#set up environment variables, these are also needed when running GPTune 
################################### 
export GPTUNEROOT=$PWD
export PATH=$GPTUNEROOT/env/bin/:$PATH
export BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
export LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/hypre-driver/
export PYTHONWARNINGS=ignore
export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export MPIRUN="$GPTUNEROOT/openmpi-4.0.1/bin/mpirun"
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.so
export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  

CC=gcc-10
FTN=gfortran-10
CPP=g++-10


# install dependencies using apt-get and virtualenv
###################################

apt-get update
apt-get install dialog apt-utils -y
apt-get install build-essential software-properties-common -y
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update
apt-get install gcc-10 g++-10 gfortran-10 -y
apt-get install libffi-dev -y
apt-get install libssl-dev -y

apt-get install libblas-dev  -y
apt-get install liblapack-dev -y
apt-get install cmake -y
apt-get install git -y
apt-get install vim -y
apt-get install autoconf automake libtool -y
apt-get install zlib1g-dev -y
apt-get install wget -y
apt-get install libsm6 -y
apt-get install libbz2-dev -y

cd $GPTUNEROOT
rm -rf Python-3.7.9
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xvf Python-3.7.9.tgz
cd Python-3.7.9
./configure --prefix=$PWD CC=$CC
make -j32
make altinstall
PY=$PWD/bin/python3.7  # this makes sure virtualenv uses the correct python version
PIP=$PWD/bin/pip3.7


cd $GPTUNEROOT
$PIP install virtualenv 
rm -rf env
$PY -m venv env
source env/bin/activate
# unalias pip  # this makes sure virtualenv install packages at its own site-packages directory
# unalias python




# manually install dependencies from cmake and make
###################################
cd $GPTUNEROOT
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
bzip2 -d openmpi-4.0.1.tar.bz2
tar -xvf openmpi-4.0.1.tar 
cd openmpi-4.0.1/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
make -j32
make install


cd $GPTUNEROOT
rm -rf scalapack-2.1.0
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
mkdir -p install
cmake .. \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DBLAS_LIBRARIES="$BLAS_LIB" \
	-DLAPACK_LIBRARIES="$LAPACK_LIB"
make -j32
make install


cd $GPTUNEROOT
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="" \
	-DCMAKE_C_FLAGS="-fopenmp" \
	-DCMAKE_Fortran_FLAGS="-fopenmp -fallow-argument-mismatch" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
	-DTPL_SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
make -j32
cp lib_gptuneclcm.so ../.
cp pdqrdriver ../



cd $GPTUNEROOT
cd examples/
rm -rf superlu_dist
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
make install -j32 

cd ../
cp $PWD/parmetis-4.0.3/build/Darwin-x86_64/libmetis/libmetis.a $PWD/parmetis-4.0.3/install/lib/.
PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
PARMETIS_LIBRARIES="$PWD/parmetis-4.0.3/install/lib/libparmetis.so"

mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRelease" \
	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn
make pzdrive_spawn

cd $GPTUNEROOT
cd examples/
rm -rf hypre
git clone https://github.com/hypre-space/hypre.git
cd hypre/src/
./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI"
make -j32
cp ../../hypre-driver/src/ij.c ./test/.
make test




# manually install dependencies from python
###################################
cd $GPTUNEROOT
python --version
pip --version
pip install --force-reinstall --upgrade -r requirements.txt
cp patches/opentuner/manipulator.py  ./env/lib/python3.7/site-packages/opentuner/search/.




cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install

cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
pip install -e .

cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install -e .






