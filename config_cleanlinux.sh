#!/bin/bash

##################################################
##################################################

export ModuleEnv='cleanlinux-unknown-openmpi-gnu'
BuildExample=1 # whether to build all examples
MPIFromSource=1 # whether to build openmpi from source

if [[ $(cat /etc/os-release | grep "PRETTY_NAME") != *"Ubuntu"* && $(cat /etc/os-release | grep "PRETTY_NAME") != *"Debian"* ]]; then
	echo "This script can only be used for Ubuntu or Debian systems"
	exit
fi

##################################################
##################################################


export GPTUNEROOT=$PWD

############### Yang's tr4 machine
if [ $ModuleEnv = 'cleanlinux-unknown-openmpi-gnu' ]; then
	
	CC=gcc-8
	FTN=gfortran-8
	CPP=g++-8

	if [[ $MPIFromSource = 1 ]]; then
		export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
		export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
		export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
		export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
		export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
		export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  		
	else

		#######################################
		#  define the following as needed
		export MPICC=
		export MPICXX=
		export MPIF90=
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
		export LIBRARY_PATH=$LIBRARY_PATH 
		export PATH=$PATH 
		########################################

		if [[ -z "$MPICC" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi will not be built from source, please set MPICC, MPICXX, MPIF90, PATH, LIBRARY_PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above. Make sure OpenMPI > 4.0.0 is used and compiled with CC=$CC, CXX=$CPP and FC=$FTN."
			exit
		fi
	fi
	export PATH=$GPTUNEROOT/env/bin/:$PATH
	export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.so
	export BLAS_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
	export LAPACK_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
	export LD_LIBRARY_PATH=$GPTUNEROOT/OpenBLAS/:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
	OPENMPFLAG=fopenmp


fi
###############





#set up environment variables, these are also needed when running GPTune 
################################### 



export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore




export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include;$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include"
export PARMETIS_LIBRARIES=$ParMETIS_DIR/lib/libparmetis.so
export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.so







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


pip install --upgrade  -r requirements.txt

# manually install dependencies from cmake and make
###################################
cd $GPTUNEROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN install -j32


if [[ $MPIFromSource = 1 ]]; then
	cd $GPTUNEROOT
	wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
	bzip2 -d openmpi-4.0.1.tar.bz2
	tar -xvf openmpi-4.0.1.tar 
	cd openmpi-4.0.1/ 
	./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
	make -j32
	make install
fi

# if openmpi, scalapack needs to be built from source
if [[ $ModuleEnv == *"openmpi"* ]]; then
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
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
cp lib_gptuneclcm.so ../.
# cp pdqrdriver ../

if [[ $BuildExample == 1 ]]; then

	cd $GPTUNEROOT/examples/SuperLU_DIST
	rm -rf superlu_dist
	git clone https://github.com/xiaoyeli/superlu_dist.git
	cd superlu_dist

	wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
	tar -xf parmetis-4.0.3.tar.gz
	cd parmetis-4.0.3/
	cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
	mkdir -p install
	make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	make install > make_parmetis_install.log 2>&1

	cd ../
	cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.
	mkdir -p build
	cd build
	rm -rf CMakeCache.txt
	rm -rf DartConfiguration.tcl
	rm -rf CTestTestfile.cmake
	rm -rf cmake_install.cmake
	rm -rf CMakeFiles
	cmake .. \
		-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
		-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
	make pddrive_spawn
	make pzdrive_spawn


	cd $GPTUNEROOT/examples/Hypre
	rm -rf hypre
	git clone https://github.com/hypre-space/hypre.git
	cd hypre/src/
	./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI" --enable-shared
	make
	cp ../../hypre-driver/src/ij.c ./test/.
	make test


	cd $GPTUNEROOT/examples/ButterflyPACK
	rm -rf ButterflyPACK
	git clone https://github.com/liuyangzhuan/ButterflyPACK.git
	cd ButterflyPACK
	git clone https://github.com/opencollab/arpack-ng.git
	cd arpack-ng
	git checkout f670e731b7077c78771eb25b48f6bf9ca47a490e
	mkdir -p build
	cd build
	cmake .. \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_INSTALL_PREFIX=. \
		-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DCMAKE_Fortran_FLAGS="" \
		-DBLAS_LIBRARIES="${BLAS_LIB}" \
		-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DMPI=ON \
		-DEXAMPLES=ON \
		-DCOVERALLS=OFF 
	make
	cd ../../
	mkdir build
	cd build
	cmake .. \
		-DCMAKE_Fortran_FLAGS="$BLAS_INC"\
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
		-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
		-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.so;$PWD/../arpack-ng/build/lib/libparpack.so"
	make -j32
	make install -j32



	cd $GPTUNEROOT/examples/STRUMPACK
	rm -rf scotch_6.1.0
	wget --no-check-certificate https://gforge.inria.fr/frs/download.php/file/38352/scotch_6.1.0.tar.gz
	tar -xf scotch_6.1.0.tar.gz
	cd ./scotch_6.1.0
	mkdir install
	cd ./src
	cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
	sed -i "s/-DSCOTCH_PTHREAD//" Makefile.inc
	sed -i "s/-DIDXSIZE64/-DIDXSIZE32/" Makefile.inc
	sed -i "s/CCD/#CCD/" Makefile.inc
	printf "CCD = $MPICC\n" >> Makefile.inc
	sed -i "s/CCP/#CCP/" Makefile.inc
	printf "CCP = $MPICC\n" >> Makefile.inc
	sed -i "s/CCS/#CCS/" Makefile.inc
	printf "CCS = $MPICC\n" >> Makefile.inc
	cat Makefile.inc
	make ptscotch 
	make prefix=../install install


	cd ../../
	rm -rf STRUMPACK
	git clone https://github.com/pghysels/STRUMPACK.git
	cd STRUMPACK
	#git checkout 959ff1115438e7fcd96b029310ed1a23375a5bf6  # head commit has compiler error, requiring fixes
	cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/. 
	cp ../STRUMPACK-driver/src/KernelRegressionMPI.py examples/. 
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


	cd $GPTUNEROOT/examples/MFEM
	cp -r $GPTUNEROOT/examples/Hypre/hypre .    # mfem requires hypre location to be here
	git clone https://github.com/mfem/mfem.git
	cd mfem
	cp ../mfem-driver/src/CMakeLists.txt ./examples/.
	cp ../mfem-driver/src/ex3p_indef.cpp ./examples/.
	rm -rf mfem-build
	mkdir mfem-build
	cd mfem-build
	cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_CXX_FLAGS="-std=c++11" \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DBUILD_SHARED_LIBS=ON \
		-DMFEM_USE_MPI=YES \
		-DCMAKE_INSTALL_PREFIX=../install \
		-DCMAKE_INSTALL_LIBDIR=../install/lib \
		-DMFEM_USE_METIS_5=YES \
		-DMFEM_USE_OPENMP=YES \
		-DMFEM_THREAD_SAFE=ON \
		-DMFEM_USE_STRUMPACK=YES \
		-DBLAS_LIBRARIES="${BLAS_LIB}" \
		-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DMETIS_DIR=${METIS_INCLUDE_DIRS} \
		-DMETIS_LIBRARIES="${METIS_LIBRARIES}" \
		-DSTRUMPACK_INCLUDE_DIRS="${STRUMPACK_DIR}/include;${METIS_INCLUDE_DIRS};${PARMETIS_INCLUDE_DIRS};${SCOTCH_DIR}/include" \
		-DSTRUMPACK_LIBRARIES="${STRUMPACK_DIR}/lib/libstrumpack.so;${ButterflyPACK_DIR}/../../../lib/libdbutterflypack.so;${ButterflyPACK_DIR}/../../../lib/libzbutterflypack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libparpack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libarpack.so;${PARMETIS_LIBRARIES};${SCALAPACK_LIB}"
	make -j32 VERBOSE=1
	make install
	make ex3p_indef

fi




cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install 
# env CC=mpicc pip install  -e .								  



cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
python setup.py build 
python setup.py install 
# env CC=mpicc pip install  -e .								  



cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
cp ../patches/autotune/problem.py autotune/.
env CC=$MPICC pip install  -e .




cd $GPTUNEROOT
cp ./patches/opentuner/manipulator.py  ./env/lib/python3.7/site-packages/opentuner/search/.

