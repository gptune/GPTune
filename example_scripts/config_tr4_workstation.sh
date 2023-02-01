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
    module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load python/gcc-9.1.0/3.7.4
	module load cmake/3.19.2
	SCALAPACK_LIB=/home/administrator/Desktop/Software/scalapack-2.0.2/build/lib/libscalapack.so
	BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
	LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
fi
###############





GPTUNEROOT=$PWD


#shopt -s expand_aliases
#alias python='python3.7'
#alias pip='pip3.7'


export PATH=$PATH:/home/administrator/.local/bin/
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-github
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so;$ParMETIS_DIR/lib/libmetis.so;$ParMETIS_DIR/lib/libGKlib.so"
export METIS_LIBRARIES="$ParMETIS_DIR/lib/libmetis.so;$ParMETIS_DIR/lib/libGKlib.so"




python --version
pip --version

if [[ -z "${GPTUNE_LITE_MODE}" ]]; then
	pip install --upgrade --user -r requirements.txt
else
	pip install --upgrade --user -r requirements_lite.txt
fi
# cp ./patches/opentuner/manipulator.py  /home/administrator/Desktop/Software/Python-3.7.4/lib/python3.7/site-packages/opentuner/search/.


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
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
make install



if [[ $BuildExample == 1 ]]; then

	cd $GPTUNEROOT/examples/SuperLU_DIST
	rm -rf superlu_dist
	git clone https://github.com/xiaoyeli/superlu_dist.git
	cd superlu_dist


	##### the following server is often down, so switch to the github repository 
	# wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
	# tar -xf parmetis-4.0.3.tar.gz
	# cd parmetis-4.0.3/
	# cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
	# mkdir -p install
	# make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	# make install > make_parmetis_install.log 2>&1
	# cd ../
	# cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	# cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


	mkdir -p $ParMETIS_DIR
	rm -f GKlib
	git clone https://github.com/KarypisLab/GKlib.git
	cd GKlib
	make config prefix=$ParMETIS_DIR
	make -j8
	make install
	sed -i "s/-DCMAKE_VERBOSE_MAKEFILE=1/-DCMAKE_VERBOSE_MAKEFILE=1 -DBUILD_SHARED_LIBS=ON/" Makefile
	make config prefix=$ParMETIS_DIR
	make -j8
	make install

	cd ../
	rm -rf METIS
	git clone https://github.com/KarypisLab/METIS.git
	cd METIS
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	make -j8
	make install
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR 
	make -j8
	make install	
	cd ../
	rm -rf ParMETIS
	git clone https://github.com/KarypisLab/ParMETIS.git
	cd ParMETIS
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	make -j8
	make install
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR
	make -j8
	make install
	cd ..

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
	make pddrive3d
	make pddrive


	cd $GPTUNEROOT/examples/Hypre
	rm -rf hypre
	git clone https://github.com/hypre-space/hypre.git
	cd hypre/src/
	git checkout v2.19.0
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
	git checkout 09fb3626cb9d7482528fce522dedad3ad9a4bc9d
	cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/sparse/. 
	cp ../STRUMPACK-driver/src/KernelRegressionMPI.py examples/dense/. 
	chmod +x examples/dense/KernelRegressionMPI.py
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
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB};$ParMETIS_DIR/lib/libGKlib.so" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

	make install -j32
	make examples -j32


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
fi


cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
cp ../patches/scikit-optimize/space.py skopt/space/.
python setup.py build 
python setup.py install --user
# env CC=mpicc pip install --user -e .								  

cd $GPTUNEROOT
rm -rf cGP
git clone https://github.com/gptune/cGP
cd cGP/
python setup.py install --user


cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
env CC=$MPICC pip install --user -e .

cd $GPTUNEROOT
rm -rf hybridMinimization
git clone https://github.com/gptune/hybridMinimization.git
cd hybridMinimization/
python setup.py install --user




