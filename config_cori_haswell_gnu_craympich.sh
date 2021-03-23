#!/bin/bash
rm -rf  ~/.cache/pip
rm -rf ~/.local/cori/
export CRAYPE_LINK_TYPE=dynamic
module load python/3.7-anaconda-2019.10
module swap cray-mpich cray-mpich/7.7.12
module unload cmake
module load cmake/3.14.4

module swap PrgEnv-intel PrgEnv-gnu
GPTUNEROOT=$PWD
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
# module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
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
export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.a


MPICC=cc
MPICXX=CC
MPIF90=ftn

#pip uninstall -r requirements.txt
#env CC=$MPICC pip install --upgrade --user -r requirements.txt
env CC=$MPICC pip install --user -r requirements.txt




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
    -DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	-DLAPACK_LIBRARIES="${LAPACK_LIB}"
make -j32 

cd $GPTUNEROOT
mkdir -p build
cd build
export CRAYPE_LINK_TYPE=dynamic
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="" \
	-DCMAKE_C_FLAGS="" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make
cp lib_gptuneclcm.so ../.
# cp pdqrdriver ../



cd $GPTUNEROOT/examples/SuperLU_DIST
rm -rf superlu_dist
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
make install > make_parmetis_install.log 2>&1

cd ../
cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a $PWD/parmetis-4.0.3/install/lib/.
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
	-DBUILD_SHARED_LIBS=OFF \
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
	-DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DMPI=ON \
	-DEXAMPLES=ON \
	-DCOVERALLS=ON 
make
cd ../../
mkdir build
cd build
cmake .. \
	-DCMAKE_Fortran_FLAGS="-I${MKLROOT}/include"\
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


cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install --user
# env CC=mpicc pip install --user -e .								  



cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
python setup.py build 
python setup.py install --user
# env CC=mpicc pip install --user -e .								  



cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
env CC=$MPICC pip install --user -e .


cp ../patches/opentuner/manipulator.py  ~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages/opentuner/search/.