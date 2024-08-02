#!/bin/bash

cd ../
export GPTUNEROOT=$PWD

##
## CHECKLIST 1.
##
## Check if the following software packages are available.
##
## gcc-13, g++-13, gfortran-13, libffi-dev, libssl-dev, cmake, git,
## autoconf, automake, libtool, zlib1g-dev, wget libsm6, libbz2-dev,
## libsqlite3-dev jq
##

##
## CHECKLIST 2.
##
## If you already have OpenMPI installed on the target platform and would like
## to use it for GPTune, define the following, as needed.
## Otherwise, please just leave it as it is, and our installation script will
## try to install OpenMPI 4.1.5 automatically from the source code.
## 

export MPICC=
export MPICXX=
export MPIF90=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH 
export PATH=$PATH 

if [[ -z "$MPICC" ]]; then
    export MPICC="$GPTUNEROOT/openmpi-4.1.5/bin/mpicc"
    export MPICXX="$GPTUNEROOT/openmpi-4.1.5/bin/mpicxx"
    export MPIF90="$GPTUNEROOT/openmpi-4.1.5/bin/mpif90"
    export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.1.5/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.1.5/lib:$LIBRARY_PATH  		
    export PATH=$PATH:$GPTUNEROOT/openmpi-4.1.5/bin
    MPIFromSource=1
fi

##
## CHECKLIST 3.
##
## If you want to use specific C, C++, Fortran compiler, define the foollowing,
## as needed.
## Otherwise, it will assume gcc-13, g++-13, and gfortran-13.
##

CC=gcc-13
CXX=g++-13
FTN=gfortran-13

##
## CHECKLIST 4.
##
## If you have python installed (e.g., with virtualenv), set the following accordingly

PyMAJOR=3
PyMINOR=9
export SITE_PACKAGE_DIR=$GPTUNEROOT/env/lib/python$PyMAJOR.$PyMINOR/site-packages


##
## INSTALLATION BEGINS
##

export PATH=$GPTUNEROOT/env/bin/:$PATH
export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.2.0/build/install/lib/libscalapack.so
export BLAS_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
export LAPACK_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
export LD_LIBRARY_PATH=$GPTUNEROOT/OpenBLAS/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.2.0/build/install/lib/:$LD_LIBRARY_PATH
OPENMPFLAG=fopenmp

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so;$ParMETIS_DIR/lib/libmetis.so"
export METIS_LIBRARIES="$ParMETIS_DIR/lib/libmetis.so"

cd $GPTUNEROOT

if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
	pip install --upgrade -r requirements.txt
else
	pip install --upgrade -r requirements_lite.txt
fi

# manually install dependencies from cmake and make
###################################
cd $GPTUNEROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CXX FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CXX FC=$FTN install -j32

if [[ $MPIFromSource = 1 ]]; then
	cd $GPTUNEROOT
	wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2
	bzip2 -d openmpi-4.1.5.tar.bz2
	tar -xvf openmpi-4.1.5.tar 
	cd openmpi-4.1.5/ 
	./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CXX F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
	make -j32
	make install
fi

# if openmpi, scalapack needs to be built from source
if [[ $ModuleEnv == *"openmpi"* ]]; then
cd $GPTUNEROOT
rm -rf scalapack-2.2.0.tgz*
wget http://www.netlib.org/scalapack/scalapack-2.2.0.tgz
tar -xf scalapack-2.2.0.tgz
cd scalapack-2.2.0
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
	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG -fallow-argument-mismatch" \
	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
make -j32
make install
fi

cd $GPTUNEROOT
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-$OPENMPFLAG" \
	-DCMAKE_C_FLAGS="-$OPENMPFLAG" \
	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG -fallow-argument-mismatch" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DGPTUNE_INSTALL_PATH=$PWD \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
make install
# cp lib_gptuneclcm.so ../.
# cp pdqrdriver ../

if [[ -z "${GPTUNE_LITE_MODE}" ]]; then

	cd $GPTUNEROOT
	rm -rf mpi4py
	git clone https://github.com/mpi4py/mpi4py.git
	cd mpi4py/
	python setup.py build --mpicc="$MPICC -shared"
	python setup.py install 
	# env CC=mpicc pip install  -e .
	#### install pygmo and its dependencies tbb, boost, pagmo from source, as pip install pygmo for python >3.8 is not working yet on some linux distributions 
	export TBB_ROOT=$GPTUNEROOT/oneTBB/build
	export pybind11_DIR=$SITE_PACKAGE_DIR/pybind11/share/cmake/pybind11
	export BOOST_ROOT=$GPTUNEROOT/boost_1_69_0/build
	export pagmo_DIR=$GPTUNEROOT/pagmo2/build/lib/cmake/pagmo
    cd $GPTUNEROOT
    rm -rf oneTBB
    git clone https://github.com/oneapi-src/oneTBB.git
    cd oneTBB
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_INSTALL_LIBDIR=$PWD/lib -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
    make -j16
    make install
    git clone https://github.com/wjakob/tbb.git
    cp tbb/include/tbb/tbb_stddef.h include/tbb/.

    cd $GPTUNEROOT
    rm -rf download
    wget -c 'http://sourceforge.net/projects/boost/files/boost/1.69.0/boost_1_69_0.tar.bz2/download'
    tar -xvf download
    cd boost_1_69_0/
    ./bootstrap.sh --prefix=$PWD/build
    ./b2 install


    cd $GPTUNEROOT
    rm -rf pagmo2
    git clone https://github.com/esa/pagmo2.git
    cd pagmo2
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_INSTALL_LIBDIR=$PWD/lib
    make -j16
    make install
    cp lib/cmake/pagmo/*.cmake . 

    cd $GPTUNEROOT
    rm -rf pygmo2
    git clone https://github.com/esa/pygmo2.git
    cd pygmo2
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DPYGMO_INSTALL_PATH="${SITE_PACKAGE_DIR}" -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -Dpagmo_DIR=${GPTUNEROOT}/pagmo2/build/ -Dpybind11_DIR=${pybind11_DIR}
    make -j16
    make install
fi
