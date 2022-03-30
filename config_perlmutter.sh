#!/bin/bash


if [[ $NERSC_HOST != "perlmutter" ]]; then
	echo "This script can only be used for perlmutter"
	exit
fi


PY_VERSION=3.9
PY_TIME=2021.11


rm -rf  ~/.cache/pip
rm -rf ~/.local/perlmutter/
rm -rf ~/.local/lib/python$PY_VERSION
module load python/$PY_VERSION-anaconda-$PY_TIME
PREFIX_PATH=~/.local/perlmutter/$PY_VERSION-anaconda-$PY_TIME/


echo $(which python) 

module unload cmake
module load cmake/3.22.0


##################################################
##################################################
machine=perlmutter
proc=gpu   # milan,gpu
mpi=craympich    # craympich
compiler=gnu   # gnu, intel	


BuildExample=1 # whether to build all examples

export ModuleEnv=$machine-$proc-$mpi-$compiler


##################################################
##################################################

echo "The ModuleEnv is $ModuleEnv"
if [ $ModuleEnv = 'perlmutter-gpu-craympich-gnu' ]; then
	export CRAYPE_LINK_TYPE=dynamic
	module swap PrgEnv-nvidia PrgEnv-gnu
	module load cudatoolkit
	GPTUNEROOT=$PWD
	BLAS_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
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
	module swap PrgEnv-nvidia PrgEnv-gnu
	GPTUNEROOT=$PWD
	BLAS_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=fopenmp
	SLU_ENABLE_CUDA=FALSE
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
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include;$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include"
export PARMETIS_LIBRARIES=$ParMETIS_DIR/lib/libparmetis.so
export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.so
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
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	-DGPTUNE_INSTALL_PATH="${PREFIX_PATH}/lib/python$PY_VERSION/site-packages"
make install
# cp lib_gptuneclcm.so ../.
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
	cp $PWD/build/Linux-x86_64/libmetis/libmetis.so $PWD/install/lib/.
	cp $PWD/metis/include/metis.h $PWD/install/include/.
	mkdir -p install_static
	make config cc=$MPICC cxx=$MPICXX prefix=$PWD/install_static
	make install > make_parmetis_install.log 2>&1
	cp $PWD/build/Linux-x86_64/libmetis/libmetis.a $PWD/install/lib/.
	cp $PWD/build/Linux-x86_64/libparmetis/libparmetis.a $PWD/install/lib/.

	cd ../
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
		-DCMAKE_CUDA_FLAGS=$SLU_CUDA_FLAG \
		-DTPL_ENABLE_CUDALIB=$SLU_ENABLE_CUDA \
		-DCMAKE_CUDA_ARCHITECTURES=80 \
		-DCMAKE_INSTALL_PREFIX=. \
	 	-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB};${CUBLAS_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
	make pddrive_spawn
	make pzdrive_spawn
	make install


	# cd $GPTUNEROOT/examples/Hypre
	# rm -rf hypre
	# git clone https://github.com/hypre-space/hypre.git
	# cd hypre/src/
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
	# 	-DCMAKE_Fortran_FLAGS="-DMPIMODULE $BLAS_INC"\
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
	# 	-DSTRUMPACK_USE_CUDA=${STRUMPACK_USE_CUDA} \
	# 	-DTPL_CUBLAS_LIBRARIES="${CUBLAS_LIB}" \
	# 	-DTPL_CUBLAS_INCLUDE_DIRS="${CUBLAS_INCLUDE}" \
	# 	-DCMAKE_CUDA_FLAGS="${STRUMPACK_CUDA_FLAGS}" \
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

# cd $GPTUNEROOT
# wget -c 'http://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.tar.bz2/download'
# tar -xvf download
# cd boost_1_68_0/
# ./bootstrap.sh --prefix=$PWD/build
# ./b2 install
# export BOOST_ROOT=$GPTUNEROOT/boost_1_68_0/stage

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
cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX_PATH -DPYGMO_INSTALL_PATH="${PREFIX_PATH}/lib/python$PY_VERSION/site-packages" -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX
make -j16
make install

cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install --prefix=$PREFIX_PATH
# env CC=mpicc pip install --user -e .								  



cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
cp ../patches/scikit-optimize/space.py skopt/space/.
python setup.py build 
python setup.py install --prefix=$PREFIX_PATH
# env CC=mpicc pip install --user -e .								  


cd $GPTUNEROOT
rm -rf cGP
git clone https://github.com/gptune/cGP
cd cGP/
python setup.py install --prefix=$PREFIX_PATH


cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
cp ../patches/autotune/problem.py autotune/.
env CC=$MPICC pip install --prefix=$PREFIX_PATH -e .


cp ../patches/opentuner/manipulator.py  $PREFIX_PATH/lib/python$PY_VERSION/site-packages/opentuner/search/.
cd $GPTUNEROOT

