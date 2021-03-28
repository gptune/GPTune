#!/bin/zsh

##################################################
##################################################
#define package version numbers from homebrew, this may need to be changed according to your system 
pythonversion=3.9.2_3
gccversion=10.2.0_4
openblasversion=0.3.13
lapackversion=3.9.0_1

export ModuleEnv='mac-intel-openmpi-gnu'
BuildExample=0 # whether to build all examples
MPIFromSource=1 # whether to build openmpi from source
##################################################
##################################################

if [[ $(uname -s) != "Darwin" ]]; then
	echo "This script can only be used for Mac OS"
	exit
fi

export GPTUNEROOT=$PWD

############### macbook
if [ $ModuleEnv = 'mac-intel-openmpi-gnu' ]; then

	export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
	export PATH=$GPTUNEROOT/env/bin/:$PATH
	export BLAS_LIB=/usr/local/Cellar/openblas/$openblasversion/lib/libblas.dylib
	export LAPACK_LIB=/usr/local/Cellar/lapack/$lapackversion/lib/liblapack.dylib
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/pygmo2/build/pygmo/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/scalapack-driver/spt/
	export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/hypre-driver/
	export PYTHONWARNINGS=ignore

	export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.dylib
	export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
	export LIBRARY_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
	export DYLD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$DYLD_LIBRARY_PATH
	export DYLD_LIBRARY_PATH=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH
	OPENMPFLAG=fopenmp
	CC=/usr/local/Cellar/gcc/$gccversion/bin/gcc-10
	FTN=/usr/local/Cellar/gcc/$gccversion/bin/gfortran-10
	CPP=/usr/local/Cellar/gcc/$gccversion/bin/g++-10

	if [[ $MPIFromSource = 1 ]]; then
		export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
		export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
		export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
		export PATH=$GPTUNEROOT/openmpi-4.0.1/bin:$PATH
		export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
		export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  
		export DYLD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib/:$DYLD_LIBRARY_PATH
	else 

		#######################################
		#  define the following as needed
		export MPICC=
		export MPICXX=
		export MPIF90=
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
		export LIBRARY_PATH=$LIBRARY_PATH 
		export PATH=$PATH 
		export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH 
		########################################

		if [[ -z "$MPICC" ]]; then
			echo "Line: ${LINENO} of ${(%):-%x}: It seems that openmpi will not be built from source, please set MPICC, MPICXX, MPIF90, PATH, LIBRARY_PATH, LD_LIBRARY_PATH, DYLD_LIBRARY_PATH for your OpenMPI build correctly above. Make sure OpenMPI > 4.0.0 is used and compiled with CC=$CC, CXX=$CPP and FC=$FTN."
			exit
		fi
	fi	

fi
###############


export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include;$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libmetis.dylib;$ParMETIS_DIR/lib/libparmetis.dylib"
export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.dylib

# install dependencies using homebrew and virtualenv
###################################
# softwareupdate --all --install --force
brew install wget
brew upgrade wget
brew install python@3.9
brew upgrade python@3.9


if [ ! -d "/usr/local/Cellar/python@3.9/$pythonversion" ] 
then
    echo "pythonversion=$pythonversion not working, change it to the correct one." 
    exit 
fi

alias python=/usr/local/Cellar/python@3.9/$pythonversion/bin/python3  # this makes sure virtualenv uses the correct python version
alias pip=/usr/local/Cellar/python@3.9/$pythonversion/bin/pip3

python -m pip install virtualenv 
rm -rf env
python -m venv env
source env/bin/activate

unalias pip  # this makes sure virtualenv install packages at its own site-packages directory
unalias python

pip install --force-reinstall cloudpickle
pip install --force-reinstall filelock
brew reinstall tbb
brew reinstall pagmo
brew reinstall pybind11

brew install gcc
brew upgrade gcc   
if [ ! -d "/usr/local/Cellar/gcc/$gccversion" ] 
then
    echo "gccversion=$gccversion not working, change it to the correct one." 
    exit 
fi


brew install openblas
brew upgrade openblas  
if [ ! -d "/usr/local/Cellar/openblas/$openblasversion" ] 
then
    echo "openblasversion=$openblasversion not working, change it to the correct one." 
    exit 
fi

brew install lapack
brew upgrade lapack   
if [ ! -d "/usr/local/Cellar/lapack/$lapackversion" ] 
then
    echo "lapackversion=$lapackversion not working, change it to the correct one." 
    exit 
fi

brew install jq


# manually install dependencies from python
###################################
cd $GPTUNEROOT
python --version
pip --version
pip install --force-reinstall --upgrade -r requirements_mac.txt
#env CC=$MPICC pip install --upgrade --user -r requirements.txt




# manually install dependencies from cmake and make
###################################
cd $GPTUNEROOT
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
bzip2 -d openmpi-4.0.1.tar.bz2
tar -xvf openmpi-4.0.1.tar 
cd openmpi-4.0.1/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
make -j8
make install


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
	mkdir -p install
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
	make -j8
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
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j8
cp lib_gptuneclcm.dylib ../.
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
	cp $PWD/parmetis-4.0.3/build/Darwin-x86_64/libmetis/libmetis.dylib $PWD/parmetis-4.0.3/install/lib/.
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
		-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
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
		-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG -fallow-argument-mismatch" \
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
		-DCMAKE_Fortran_FLAGS=" -fallow-argument-mismatch"\
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
		-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.dylib;$PWD/../arpack-ng/build/lib/libparpack.dylib"
	make -j8
	make install -j8



	cd $GPTUNEROOT/examples/STRUMPACK
	rm -rf scotch_6.1.0
	wget --no-check-certificate https://gforge.inria.fr/frs/download.php/file/38352/scotch_6.1.0.tar.gz
	tar -xf scotch_6.1.0.tar.gz
	cd ./scotch_6.1.0
	mkdir install
	cd ./src
	cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
	sed -i "" "s/-DSCOTCH_PTHREAD//" Makefile.inc
	sed -i "" "s/-lrt//" Makefile.inc
	sed -i "" "s/-DIDXSIZE64/-DIDXSIZE32/" Makefile.inc
	sed -i "" "s/CCD/#CCD/" Makefile.inc
	printf "CCD = $MPICC\n" >> Makefile.inc
	sed -i "" "s/CCP/#CCP/" Makefile.inc
	printf "CCP = $MPICC\n" >> Makefile.inc
	sed -i "" "s/CCS/#CCS/" Makefile.inc
	printf "CCS = $MPICC\n" >> Makefile.inc
	cat Makefile.inc
	cp ../../../../patches/ptscotch/common.* libscotch/.
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
		-DCMAKE_Fortran_FLAGS=" -fallow-argument-mismatch" \
		-DSTRUMPACK_TASK_TIMERS=ON \
		-DTPL_ENABLE_SCOTCH=ON \
		-DTPL_ENABLE_ZFP=OFF \
		-DTPL_ENABLE_PTSCOTCH=ON \
		-DTPL_ENABLE_PARMETIS=ON \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

	make install -j8
	make examples -j8


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
		-DCMAKE_Fortran_FLAGS=" -fallow-argument-mismatch" \
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
		-DSTRUMPACK_LIBRARIES="${STRUMPACK_DIR}/lib/libstrumpack.dylib;${ButterflyPACK_DIR}/../../../lib/libdbutterflypack.dylib;${ButterflyPACK_DIR}/../../../lib/libzbutterflypack.dylib;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libparpack.dylib;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libarpack.dylib;${PARMETIS_LIBRARIES};${SCALAPACK_LIB}"
	make -j8 VERBOSE=1
	make install
	make ex3p_indef

fi

# # pip install pygmo doesn't work, build from source, note that it's built with clang, as brew pagmo uses clang (I haven't figured out how to install boost with gnu on mac os), this may cause segfault at the search phase
cd $GPTUNEROOT
rm -rf pygmo2
git clone https://github.com/esa/pygmo2.git
cd pygmo2
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=. -DPYTHON_EXECUTABLE:FILEPATH=python -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 
make -j8


cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install
# env CC=mpicc pip install  -e .								  



cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
cp ../patches/autotune/problem.py autotune/.
pip install -e .



cd $GPTUNEROOT
rm -rf GPy
git clone https://github.com/SheffieldML/GPy.git
cd GPy
cp ../patches/GPy/coregionalize.py ./GPy/kern/src/.
cp ../patches/GPy/stationary.py ./GPy/kern/src/.
cp ../patches/GPy/choleskies.py ./GPy/util/.
LDSHARED="$MPICC -shared" CC=$MPICC python setup.py build_ext --inplace
python setup.py install



cd $GPTUNEROOT
cp patches/opentuner/manipulator.py  ./env/lib/python3.9/site-packages/opentuner/search/.

