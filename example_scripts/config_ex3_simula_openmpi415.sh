#!/bin/bash

##################################################
##################################################
cd ..
export ModuleEnv='ex3-xeongold16q-openmpi-gnu'
BuildExample=0 # whether to build all examples
MPIFromSource=0 # whether to build openmpi from source
PYTHONFromSource=0 # whether to build python from source

if [[ $(cat /etc/os-release | grep "PRETTY_NAME") != *"Ubuntu"* && $(cat /etc/os-release | grep "PRETTY_NAME") != *"Debian"* ]]; then
	echo "This script can only be used for Ubuntu or Debian systems"
	exit
fi

##################################################
##################################################


export GPTUNEROOT=$PWD

############### Yang's tr4 machine
if [ $ModuleEnv = 'ex3-xeongold16q-openmpi-gnu' ]; then

        module load cmake/gcc/3.26.4
        # module load python37
      module load openblas/dynamic/0.3.7
        # module load openblas/dynamic/0.3.23
        module load openmpi/gcc/64/4.1.5
        module load jq/1.6
        module load scalapack/gcc/2.0.2
        # module load openmpi/gcc/64/4.1.4
        module load metis/gcc/5.1.0
        module load parmetis/gcc/4.0.3
        module load scotch/gcc/6.0.7
        module load slurm/20.02.7

	CC=gcc
	FTN=gfortran
	CPP=g++

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
		export MPICC=mpicc
		export MPICXX=mpicxx
		export MPIF90=mpif90
		export LD_LIBRARY_PATH=/cm/shared/apps/openmpi/gcc/64/4.1.5/lib:$LD_LIBRARY_PATH
		export LIBRARY_PATH=/cm/shared/apps/openmpi/gcc/64/4.1.5/lib:$LIBRARY_PATH 
		export PATH=$PATH:/cm/shared/apps/openmpi/gcc/64/4.1.5/bin 
		########################################

		if [[ -z "$MPICC" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi will not be built from source, please set MPICC, MPICXX, MPIF90, PATH, LIBRARY_PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above. Make sure OpenMPI > 4.0.0 is used and compiled with CC=$CC, CXX=$CPP and FC=$FTN."
			exit
		fi
	fi
	export PATH=$GPTUNEROOT/env/bin/:$PATH
	export SCALAPACK_LIB=/cm/shared/apps/scalapack/gcc/2.0.2/lib/libscalapack.so
	export LD_LIBRARY_PATH=/cm/shared/apps/scalapack/gcc/2.0.2/lib/:$LD_LIBRARY_PATH
	# export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.so
	# export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH

	export BLAS_LIB=/cm/shared/apps/openblas/0.3.7/lib/libopenblas.so
	export LAPACK_LIB=/cm/shared/apps/openblas/0.3.7/lib/libopenblas.so
	export LD_LIBRARY_PATH=/cm/shared/apps/openblas/0.3.7/lib/:$LD_LIBRARY_PATH


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


export SCOTCH_DIR=/cm/shared/apps/scotch/gcc/6.0.7/
export ParMETIS_DIR=/cm/shared/apps/parmetis/gcc/4.0.3/
export METIS_DIR=/cm/shared/apps/metis/gcc/5.1.0/
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$METIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so"
export METIS_LIBRARIES="$METIS_DIR/lib/libmetis.so"


if [[ $PYTHONFromSource = 1 ]]; then
	PyMAJOR=3
	PyMINOR=7
	PyPATCH=9
	cd $GPTUNEROOT
	rm -rf Python-$PyMAJOR.$PyMINOR.$PyPATCH
	wget https://www.python.org/ftp/python/$PyMAJOR.$PyMINOR.$PyPATCH/Python-$PyMAJOR.$PyMINOR.$PyPATCH.tgz
	tar -xvf Python-$PyMAJOR.$PyMINOR.$PyPATCH.tgz
	cd Python-$PyMAJOR.$PyMINOR.$PyPATCH
	./configure --prefix=$PWD CC=$CC
	make -j32
	make altinstall
	PY=$PWD/bin/python$PyMAJOR.$PyMINOR  # this makes sure virtualenv uses the correct python version
	PIP=$PWD/bin/pip$PyMAJOR.$PyMINOR
else
	PyMAJOR=3 # set the correct python versions and path according to your system
	PyMINOR=8
	PyPATCH=16
	PY=python3.8  # this makes sure virtualenv uses the correct python version
	PIP=pip
	alias python=$PY
fi

cd $GPTUNEROOT
rm -rf env
$PY -m venv env
source env/bin/activate

if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
	pip install --upgrade -r requirements.txt
else
	pip install --upgrade -r requirements_lite.txt
fi
cd $GPTUNEROOT

# # if openmpi, scalapack needs to be built from source
# if [[ $ModuleEnv == *"openmpi"* ]]; then
# cd $GPTUNEROOT
# rm -rf scalapack-2.1.0.tgz*
# wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
# tar -xf scalapack-2.1.0.tgz
# cd scalapack-2.1.0
# rm -rf build
# mkdir -p build
# cd build
# cmake .. \
# 	-DBUILD_SHARED_LIBS=ON \
# 	-DCMAKE_C_COMPILER=$MPICC \
# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
# 	-DCMAKE_INSTALL_PREFIX=. \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_INSTALL_PREFIX=./install \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG " \
# 	-DBLAS_LIBRARIES="${BLAS_LIB}" \
# 	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
# make -j32
# make install
# fi


cd $GPTUNEROOT
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DCMAKE_CXX_FLAGS="-$OPENMPFLAG" \
	-DCMAKE_C_FLAGS="-$OPENMPFLAG" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DGPTUNE_INSTALL_PATH=./env/lib/python3.8/site-packages/ \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
make install


if [[ $BuildExample == 1 ]]; then

	cd $GPTUNEROOT/examples/SuperLU_DIST
	rm -rf superlu_dist
	git clone https://github.com/xiaoyeli/superlu_dist.git
	cd superlu_dist

	rm -rf build
	mkdir -p build
	cd build
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


	# cd $GPTUNEROOT/examples/Hypre
	# rm -rf hypre
	# git clone https://github.com/hypre-space/hypre.git
	# cd hypre/src/
	# git checkout v2.19.0
	# ./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI" --enable-shared
	# make
	# cp ../../hypre-driver/src/ij.c ./test/.
	# make test


	# cd $GPTUNEROOT/examples/ButterflyPACK
	# rm -rf ButterflyPACK
	# git clone https://github.com/liuyangzhuan/ButterflyPACK.git
	# cd ButterflyPACK
	# git clone https://github.com/opencollab/arpack-ng.git
	# cd arpack-ng
	# git checkout f670e731b7077c78771eb25b48f6bf9ca47a490e
	# mkdir -p build
	# cd build
	# cmake .. \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DCMAKE_C_COMPILER=$MPICC \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DCMAKE_INSTALL_PREFIX=. \
	# 	-DCMAKE_INSTALL_LIBDIR=./lib \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	# 	-DCMAKE_Fortran_FLAGS="" \
	# 	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DMPI=ON \
	# 	-DEXAMPLES=ON \
	# 	-DCOVERALLS=OFF 
	# make
	# cd ../../
	# mkdir build
	# cd build
	# cmake .. \
	# 	-DCMAKE_Fortran_FLAGS="$BLAS_INC "\
	# 	-DCMAKE_CXX_FLAGS="" \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_C_COMPILER=$MPICC \
	# 	-DCMAKE_INSTALL_PREFIX=. \
	# 	-DCMAKE_INSTALL_LIBDIR=./lib \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	# 	-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.so;$PWD/../arpack-ng/build/lib/libparpack.so"
	# make -j32
	# make install -j32



	# cd $GPTUNEROOT/examples/STRUMPACK
	# rm -rf STRUMPACK
	# git clone https://github.com/pghysels/STRUMPACK.git
	# cd STRUMPACK
	# #git checkout 959ff1115438e7fcd96b029310ed1a23375a5bf6  # head commit has compiler error, requiring fixes
	# git checkout 09fb3626cb9d7482528fce522dedad3ad9a4bc9d
	# cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/sparse/. 
	# cp ../STRUMPACK-driver/src/KernelRegressionMPI.py examples/dense/. 
	# chmod +x examples/dense/KernelRegressionMPI.py
	# mkdir build
	# cd build

	# cmake ../ \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_INSTALL_PREFIX=../install \
	# 	-DCMAKE_INSTALL_LIBDIR=../install/lib \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_C_COMPILER=$MPICC \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DSTRUMPACK_COUNT_FLOPS=ON \
	# 	-DSTRUMPACK_TASK_TIMERS=ON \
	# 	-DTPL_ENABLE_SCOTCH=ON \
	# 	-DTPL_ENABLE_ZFP=OFF \
	# 	-DTPL_ENABLE_PTSCOTCH=ON \
	# 	-DTPL_ENABLE_PARMETIS=ON \
	# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

	# make install -j32
	# make examples -j32


	# cd $GPTUNEROOT/examples/MFEM
	# cp -r $GPTUNEROOT/examples/Hypre/hypre .    # mfem requires hypre location to be here
	# git clone https://github.com/mfem/mfem.git
	# cd mfem
	# cp ../mfem-driver/src/CMakeLists.txt ./examples/.
	# cp ../mfem-driver/src/ex3p_indef.cpp ./examples/.
	# rm -rf mfem-build
	# mkdir mfem-build
	# cd mfem-build
	# cmake .. \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_CXX_FLAGS="-std=c++11" \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DMFEM_USE_MPI=YES \
	# 	-DCMAKE_INSTALL_PREFIX=../install \
	# 	-DCMAKE_INSTALL_LIBDIR=../install/lib \
	# 	-DMFEM_USE_METIS_5=YES \
	# 	-DMFEM_USE_OPENMP=YES \
	# 	-DMFEM_THREAD_SAFE=ON \
	# 	-DMFEM_USE_STRUMPACK=YES \
	# 	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DMETIS_DIR=${METIS_INCLUDE_DIRS} \
	# 	-DMETIS_LIBRARIES="${METIS_LIBRARIES}" \
	# 	-DSTRUMPACK_INCLUDE_DIRS="${STRUMPACK_DIR}/include;${METIS_INCLUDE_DIRS};${PARMETIS_INCLUDE_DIRS};${SCOTCH_DIR}/include" \
	# 	-DSTRUMPACK_LIBRARIES="${STRUMPACK_DIR}/lib/libstrumpack.so;${ButterflyPACK_DIR}/../../../lib/libdbutterflypack.so;${ButterflyPACK_DIR}/../../../lib/libzbutterflypack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libparpack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libarpack.so;${PARMETIS_LIBRARIES};${SCALAPACK_LIB}"
	# make -j32 VERBOSE=1
	# make install
	# make ex3p_indef

fi


if [[ -z "${GPTUNE_LITE_MODE}" ]]; then

	cd $GPTUNEROOT
	rm -rf mpi4py
	git clone https://github.com/mpi4py/mpi4py.git
	cd mpi4py/
	python setup.py build --mpicc="$MPICC -shared"
	python setup.py install 
	# env CC=mpicc pip install  -e .
fi



