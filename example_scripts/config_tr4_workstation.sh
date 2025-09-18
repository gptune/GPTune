#!/bin/bash

rm -rf ~/.local/lib
cd ..

##################################################
##################################################

export ModuleEnv='tr4-workstation-AMD1950X-openmpi-gnu'
BuildExample=0 # whether to build all examples

##################################################
##################################################

if [[ $(hostname -s) != "tr4-workstation" ]]; then
	echo "This script can only be used for tr4-workstation"
	exit
fi

############### Yang's tr4 machine
if [ $ModuleEnv = 'tr4-workstation-AMD1950X-openmpi-gnu' ]; then
	
	module purge
	module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load cmake/3.19.2
	module load python/gcc-9.1.0/3.7.4
	SCALAPACK_LIB=/home/administrator/Desktop/Software/scalapack-2.0.2/build/lib/libscalapack.so


	# module purge
	# module load gcc/9.1.0
    # module load openmpi/gcc-9.1.0/4.0.1
    # module load scalapack-netlib/gcc-9.1.0/2.0.2
    # module load cmake/3.19.2	
	# module load python/gcc-9.1.0/3.8.4
	# shopt -s expand_aliases
	# alias python='python3.8'
	# alias pip='pip3.8'


    ################## the following works for the instalation, but at runtime mpi spawn failed, maybe the openmpi installation is not correct
	# module purge
	# module load gcc/13.1.0
    # module load openmpi/gcc-13.1.0/4.0.1
    # module load scalapack-netlib/gcc-13.1.0/2.2.0
    # module load cmake/3.19.2	
	# module load python/gcc-13.1.0/3.12.4
	

	SCALAPACK_LIB=${SCALAPACK_LIB_DIR}/libscalapack.so
	BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
	LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
fi
###############





GPTUNEROOT=$PWD


export PATH=$PATH:/home/administrator/.local/bin/
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


module list

python --version
pip --version

if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
	pip install --upgrade --user -r requirements.txt
else
	pip install --upgrade --user -r requirements_lite.txt
fi


cd $GPTUNEROOT
rm -rf build
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
	-DGPTUNE_INSTALL_PATH=$PWD \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	-DGPTUNE_INSTALL_PATH="${SITE_PACKAGE_DIR}"
make -j32
make install



if [[ $BuildExample == 1 ]]; then

	# cd $GPTUNEROOT/examples/SuperLU_DIST
	# rm -rf superlu_dist
	# git clone https://github.com/xiaoyeli/superlu_dist.git
	# cd superlu_dist
	# # git checkout gpu_trisolve_new

	# #### the following server is often down, so switch to the github repository 
	# wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/parmetis/4.0.3-4/parmetis_4.0.3.orig.tar.gz
	# tar -xf parmetis_4.0.3.orig.tar.gz
	# cd parmetis-4.0.3/
	# cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
	# mkdir -p install
	# make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	# make install > make_parmetis_install.log 2>&1
	# cd ../
	# cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	# cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


	# mkdir -p $ParMETIS_DIR
	# rm -f GKlib
	# git clone https://github.com/KarypisLab/GKlib.git
	# cd GKlib
	# make config prefix=$ParMETIS_DIR
	# make -j8
	# make install
	# sed -i "s/-DCMAKE_VERBOSE_MAKEFILE=1/-DCMAKE_VERBOSE_MAKEFILE=1 -DBUILD_SHARED_LIBS=ON/" Makefile
	# make config prefix=$ParMETIS_DIR
	# make -j8
	# make install

	# cd ../
	# rm -rf METIS
	# git clone https://github.com/KarypisLab/METIS.git
	# cd METIS
	# make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	# make -j8
	# make install
	# make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR 
	# make -j8
	# make install	
	# cd ../
	# rm -rf ParMETIS
	# git clone https://github.com/KarypisLab/ParMETIS.git
	# cd ParMETIS
	# make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	# make -j8
	# make install
	# make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR
	# make -j8
	# make install
	# cd ..

	# mkdir -p build
	# cd build
	# rm -rf CMakeCache.txt
	# rm -rf DartConfiguration.tcl
	# rm -rf CTestTestfile.cmake
	# rm -rf cmake_install.cmake
	# rm -rf CMakeFiles
	# cmake .. \
	# 	-DCMAKE_CXX_FLAGS="-std=c++11 -DAdd_" \
	# 	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_C_COMPILER=$MPICC \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	# 	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
	# make pddrive_spawn
	# make pzdrive_spawn
	# make pddrive3d
	# make pddrive


	cd $GPTUNEROOT/examples/Hypre
	rm -rf hypre
	git clone https://github.com/hypre-space/hypre.git
	cd hypre/src/
	git checkout v2.19.0
	./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI" --enable-shared
	make
	cp ../../hypre-driver/src/ij.c ./test/.
	make test


	cd $GPTUNEROOT/examples/IMPACT-Z
	rm -rf IMPACT-Z
	git clone https://github.com/impact-lbl/IMPACT-Z.git
	cd IMPACT-Z
	git checkout f98eedd2afe8b7e9f20bb72831496b66def334b7  # the Jun 2021 commit that GPTune was able to run
	cp ../impact-z-driver/*.f90 ./src/Contrl/.
	mkdir -p build 
	cd build
	cmake ../src -DUSE_MPI=ON -DCMAKE_Fortran_COMPILER=$MPIF90 -DCMAKE_BUILD_TYPE=Release 
	make
	# mpirun -n 4 ./ImpactZexe-mpi 0 0 0 0 0


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
	# 	-DCMAKE_Fortran_FLAGS="$BLAS_INC"\
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
	# rm -rf scotch_6.1.0
	# wget --no-check-certificate https://gforge.inria.fr/frs/download.php/file/38352/scotch_6.1.0.tar.gz
	# tar -xf scotch_6.1.0.tar.gz
	# cd ./scotch_6.1.0
	# mkdir install
	# cd ./src
	# cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
	# sed -i "s/-DSCOTCH_PTHREAD//" Makefile.inc
	# sed -i "s/-DIDXSIZE64/-DIDXSIZE32/" Makefile.inc
	# sed -i "s/CCD/#CCD/" Makefile.inc
	# printf "CCD = $MPICC\n" >> Makefile.inc
	# sed -i "s/CCP/#CCP/" Makefile.inc
	# printf "CCP = $MPICC\n" >> Makefile.inc
	# sed -i "s/CCS/#CCS/" Makefile.inc
	# printf "CCS = $MPICC\n" >> Makefile.inc
	# cat Makefile.inc
	# make ptscotch 
	# make prefix=../install install


	# cd ../../
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
	# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB};$ParMETIS_DIR/lib/libGKlib.so" \
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


	# cd $GPTUNEROOT/examples/heffte_RCI
	# rm -rf heffte
	# git clone https://bitbucket.org/icl/heffte.git
	# cd heffte
	# mkdir build
	# cd build
	# # ignoring the MKL, FFTW, and CUDA dependencies for now 
	# cmake -DHeffte_ENABLE_MKL=OFF -DHeffte_ENABLE_FFTW=OFF -DHeffte_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE="-O3" ..
	# make -j8

fi


if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
	cd $GPTUNEROOT
	rm -rf mpi4py
	git clone https://github.com/mpi4py/mpi4py.git
	cd mpi4py/
	python setup.py build --mpicc="$MPICC -shared"
	python setup.py install --user
	# env CC=mpicc pip install --user -e .		


	version=$(python --version 2>&1)
	PyMINOR=$(echo "$version" | grep -oP 'Python \K[0-9]+\.[0-9]+' | cut -d. -f2)

	if [ "$PyMINOR" -gt 8 ]; then
		#### install pygmo and its dependencies tbb, boost, pagmo from source, as pip install pygmo for python >3.8 is not working yet on some linux distributions. Otherwise, one can use requirement.txt to install pygmo.   
		
		cd $GPTUNEROOT
		export TBB_ROOT=$GPTUNEROOT/oneTBB/build
		export pybind11_DIR=$SITE_PACKAGE_DIR/pybind11/share/cmake/pybind11
		export Boost_DIR=$GPTUNEROOT/boost_1_78_0/build
		export pagmo_DIR=$GPTUNEROOT/pagmo2/build/lib/cmake/pagmo
		cd $GPTUNEROOT
		rm -rf oneTBB
		git clone https://github.com/oneapi-src/oneTBB.git
		cd oneTBB
		mkdir build
		cd build
		cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_INSTALL_LIBDIR=$PWD/lib -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
		make -j
		make install
		git clone https://github.com/wjakob/tbb.git
		cp tbb/include/tbb/tbb_stddef.h include/tbb/.

		cd $GPTUNEROOT
		rm -rf download
		wget -c 'http://sourceforge.net/projects/boost/files/boost/1.78.0/boost_1_78_0.tar.bz2/download'
		tar -xvf download
		cd boost_1_78_0/
		./bootstrap.sh --prefix=$PWD/build
		./b2 install

		cd $GPTUNEROOT
		rm -rf pagmo2
		git clone https://github.com/esa/pagmo2.git
		cd pagmo2
		mkdir build
		cd build
		cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_INSTALL_LIBDIR=$PWD/lib
		make -j
		make install
		cp lib/cmake/pagmo/*.cmake . 

		cd $GPTUNEROOT
		rm -rf pygmo2
		git clone https://github.com/esa/pygmo2.git
		cd pygmo2
		mkdir build
		cd build
		cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DPYGMO_INSTALL_PATH="${SITE_PACKAGE_DIR}" -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -Dpagmo_DIR=${GPTUNEROOT}/pagmo2/build/ -Dpybind11_DIR=${pybind11_DIR}
		make -j
		make install
	fi


fi




