#!/bin/bash

cd ..

if [[ $NERSC_HOST != "perlmutter" ]]; then
	echo "This script can only be used for perlmutter"
	exit
fi

module load cpe/23.03 

PY_VERSION=3.9
PY_TIME=2021.11


rm -rf  ~/.cache/pip
rm -rf ~/.local/perlmutter/
rm -rf ~/.local/lib/python$PY_VERSION
module load python/$PY_VERSION-anaconda-$PY_TIME
PREFIX_PATH=~/.local/perlmutter/$PY_VERSION-anaconda-$PY_TIME/


echo $(which python) 

module unload cmake
module load cmake


##################################################
##################################################
machine=perlmutter
proc=milan   # milan,gpu
mpi=openmpi    # craympich, openmpi
compiler=gnu   # gnu, intel	


BuildExample=1 # whether to build all examples

export ModuleEnv=$machine-$proc-$mpi-$compiler


##################################################
##################################################

echo "The ModuleEnv is $ModuleEnv"
if [ $ModuleEnv = 'perlmutter-gpu-craympich-gnu' ]; then
	export CRAYPE_LINK_TYPE=dynamic
	module load PrgEnv-gnu
	module load cudatoolkit
	GPTUNEROOT=$PWD
	BLAS_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=fopenmp
	SLU_ENABLE_CUDA=TRUE
	SLU_CUDA_FLAG="-I${MPICH_DIR}/include"
	STRUMPACK_USE_CUDA=ON
	STRUMPACK_CUDA_FLAGS="-I${MPICH_DIR}/include"
# fi 

elif [ $ModuleEnv = 'perlmutter-milan-craympich-gnu' ]; then
	export CRAYPE_LINK_TYPE=dynamic
	module load PrgEnv-gnu
	GPTUNEROOT=$PWD
	BLAS_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_gnu_82_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=fopenmp
	SLU_ENABLE_CUDA=FALSE
# fi 

elif [ $ModuleEnv = 'perlmutter-gpu-openmpi-gnu' ]; then
	module use /global/common/software/m3169/perlmutter/modulefiles
	export CRAYPE_LINK_TYPE=dynamic
    module load PrgEnv-gnu
	module unload cray-libsci
	module unload cray-mpich
	module unload openmpi
	module load openmpi
	module unload darshan
	module load cudatoolkit

	GPTUNEROOT=$PWD
    BLAS_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
    LAPACK_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"
	MPICC=mpicc
	MPICXX=mpiCC
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
	SLU_ENABLE_CUDA=TRUE
	SLU_CUDA_FLAG="-I${OMPI_DIR}/include"
	STRUMPACK_USE_CUDA=ON
	STRUMPACK_CUDA_FLAGS="-I${OMPI_DIR}/include"
	# export UCX_NET_DEVICES=mlx5_0:1
# fi 

elif [ $ModuleEnv = 'perlmutter-milan-openmpi-gnu' ]; then
	module use /global/common/software/m3169/perlmutter/modulefiles
	export CRAYPE_LINK_TYPE=dynamic
    module load PrgEnv-gnu
	module unload cray-libsci
	module unload cray-mpich
	module unload openmpi
	module load openmpi
	module unload darshan

	GPTUNEROOT=$PWD
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMPI_DIR}/lib/
    BLAS_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
    LAPACK_LIB=$GPTUNEROOT/OpenBLAS/libopenblas.so
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"
	MPICC=mpicc
	MPICXX=mpiCC
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
	SLU_ENABLE_CUDA=FALSE
	# export UCX_NET_DEVICES=mlx5_0:1
# fi 

else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi 

export PYTHONPATH=~/.local/perlmutter/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
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
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so;$ParMETIS_DIR/lib/libmetis.so"
export METIS_LIBRARIES="$ParMETIS_DIR/lib/libmetis.so"

export TBB_ROOT=$GPTUNEROOT/oneTBB/build
export pybind11_DIR=$PREFIX_PATH/lib/python$PY_VERSION/site-packages/pybind11/share/cmake/pybind11
# export BOOST_ROOT=/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/boost_1_68_0/build
export BOOST_ROOT=/global/common/software/nersc/pm-2021q4/spack/cray-sles15-zen3/boost-1.78.0-ixcb3d5/
export pagmo_DIR=$GPTUNEROOT/pagmo2/build/lib/cmake/pagmo








if [[ $ModuleEnv == *"intel"* ]]; then
	rm -rf GPy
	git clone https://github.com/SheffieldML/GPy.git
	cd GPy
	cp ../patches/GPy/coregionalize.py ./GPy/kern/src/.
	cp ../patches/GPy/stationary.py ./GPy/kern/src/.
	cp ../patches/GPy/choleskies.py ./GPy/util/.
	LDSHARED="$MPICC -shared" CC=$MPICC python setup.py build_ext --inplace
	python setup.py install --prefix=$PREFIX_PATH
	cd $GPTUNEROOT
	env CC=$MPICC pip install --prefix=$PREFIX_PATH -r requirements_intel.txt
else 
	env CC=$MPICC pip install --prefix=$PREFIX_PATH -r requirements_perlmutter.txt
fi
# cp ./patches/opentuner/manipulator.py  $PREFIX_PATH/lib/python$PY_VERSION/site-packages/opentuner/search/.

# if openmpi, scalapack needs to be built from source
if [[ $ModuleEnv == *"openmpi"* ]]; then

	cd $GPTUNEROOT
	git clone https://github.com/xianyi/OpenBLAS
	cd OpenBLAS
	make PREFIX=. CC=$MPICC CXX=$MPICXX FC=$MPIF90 -j32
	make PREFIX=. CC=$MPICC CXX=$MPICXX FC=$MPIF90 install -j32

	cd $GPTUNEROOT
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
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG -fallow-argument-mismatch" \
		-DBLAS_LIBRARIES="${BLAS_LIB}" \
		-DLAPACK_LIBRARIES="${LAPACK_LIB}"
	make -j32
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
	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG -fallow-argument-mismatch" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	-DGPTUNE_INSTALL_PATH="${PREFIX_PATH}/lib/python$PY_VERSION/site-packages"
make install


if [[ $BuildExample == 1 ]]; then

	cd $GPTUNEROOT/examples/SuperLU_DIST
	rm -rf superlu_dist
	git clone https://github.com/xiaoyeli/superlu_dist.git
	cd superlu_dist

	#### the following server is often down, so switch to the github repository 
	# wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
	# tar -xf parmetis-4.0.3.tar.gz
	wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/parmetis/4.0.3-4/parmetis_4.0.3.orig.tar.gz
	tar -xf parmetis_4.0.3.orig.tar.gz
	cd parmetis-4.0.3/
	cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
	mkdir -p install
	make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	make install > make_parmetis_install.log 2>&1
	cd ../
	cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


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
	# cd ../

	mkdir -p build
	cd build
	rm -rf CMakeCache.txt
	rm -rf DartConfiguration.tcl
	rm -rf CTestTestfile.cmake
	rm -rf cmake_install.cmake
	rm -rf CMakeFiles
	cmake .. \
		-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
		-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_CUDA_FLAGS=$SLU_CUDA_FLAG \
		-DTPL_ENABLE_CUDALIB=$SLU_ENABLE_CUDA \
		-DCMAKE_CUDA_ARCHITECTURES=80 \
		-DCMAKE_INSTALL_PREFIX=. \
	 	-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB};${CUBLAS_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_INCLUDE_DIRS}" \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
	make pddrive_spawn
	make pzdrive_spawn
	make pddrive3d
	make install


	# cd $GPTUNEROOT/examples/Hypre
	# rm -rf hypre
	# git clone https://github.com/hypre-space/hypre.git
	# cd hypre/src/
	# git checkout v2.19.0
	# ./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI" --enable-shared
	# make
	# cp ../../hypre-driver/src/ij.c ./test/.
	# make test


	cd $GPTUNEROOT/examples/ButterflyPACK
	rm -rf ButterflyPACK
	git clone https://github.com/liuyangzhuan/ButterflyPACK.git
	cd ButterflyPACK
	git clone https://github.com/opencollab/arpack-ng.git
	cd arpack-ng
	git checkout f670e731b7077c78771eb25b48f6bf9ca47a490e
	cp ../patches/PARPACK/pzneupd.f ./PARPACK/SRC/MPI/. 
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
		-DCMAKE_Fortran_FLAGS="-DMPIMODULE $BLAS_INC -fallow-argument-mismatch" \
		-DCMAKE_CXX_FLAGS="" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_INSTALL_PREFIX=. \
		-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
		-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.so;$PWD/../arpack-ng/build/lib/libparpack.so"
	make -j32
	make install -j32



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
	# 	-DSTRUMPACK_USE_CUDA=${STRUMPACK_USE_CUDA} \
	# 	-DTPL_CUBLAS_LIBRARIES="${CUBLAS_LIB}" \
	# 	-DTPL_CUBLAS_INCLUDE_DIRS="${CUBLAS_INCLUDE}" \
	# 	-DCMAKE_CUDA_FLAGS="${STRUMPACK_CUDA_FLAGS}" \
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


	# cd $GPTUNEROOT/examples/IMPACT-Z
	# rm -rf IMPACT-Z
	# git clone https://github.com/impact-lbl/IMPACT-Z.git
	# cd IMPACT-Z
	# cp ../impact-z-driver/*.f90 ./src/Contrl/.
	# mkdir -p build 
	# cd build
	# cmake ../src -DUSE_MPI=ON -DCMAKE_Fortran_COMPILER=$MPIF90 -DCMAKE_BUILD_TYPE=Release
	# make
	# # mpirun -n 4 ./ImpactZexe-mpi 0 0 0 0 0

    cd $GPTUNEROOT/examples/heffte_RCI
    rm -rf heffte
    git clone https://github.com/icl-utk-edu/heffte.git
    cd heffte
    mkdir build
    cd build
    # ignoring the MKL, FFTW, and CUDA dependencies for now
    cmake -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_C_COMPILER=$MPICC -DCMAKE_Fortran_COMPILER=$MPIF90 -DHeffte_ENABLE_MKL=OFF -DHeffte_ENABLE_FFTW=OFF -DHeffte_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE="-O3" ..
    make -j8
fi




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
export BOOST_ROOT=$GPTUNEROOT/boost_1_69_0/build

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
cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX_PATH -DPYGMO_INSTALL_PATH="${PREFIX_PATH}/lib/python$PY_VERSION/site-packages" -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -Dpagmo_DIR=${GPTUNEROOT}/pagmo2/build/ -Dpybind11_DIR=${PREFIX_PATH}/lib/python$PY_VERSION/site-packages/pybind11/share/cmake/pybind11
make -j16
make install

cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install --prefix=$PREFIX_PATH
# env CC=mpicc pip install --user -e .								  



# cd $GPTUNEROOT
# rm -rf scikit-optimize
# git clone https://github.com/scikit-optimize/scikit-optimize.git
# cd scikit-optimize/
# cp ../patches/scikit-optimize/space.py skopt/space/.
# python setup.py build 
# python setup.py install --prefix=$PREFIX_PATH
# # env CC=mpicc pip install --user -e .								  


# cd $GPTUNEROOT
# rm -rf cGP
# git clone https://github.com/gptune/cGP
# cd cGP/
# python setup.py install --prefix=$PREFIX_PATH


# cd $GPTUNEROOT
# rm -rf autotune
# git clone https://github.com/ytopt-team/autotune.git
# cd autotune/
# # cp ../patches/autotune/problem.py autotune/.
# env CC=$MPICC pip install --prefix=$PREFIX_PATH -e .

# cd $GPTUNEROOT
# rm -rf hybridMinimization
# git clone https://github.com/gptune/hybridMinimization.git
# cd hybridMinimization/
# python setup.py install --prefix=$PREFIX_PATH


cd $GPTUNEROOT

