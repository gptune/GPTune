#!/bin/bash
rm -rf  ~/.cache/pip
rm -rf ~/.local/cori/
  
module load python/3.7-anaconda-2019.10
module unload cray-mpich
module unload cmake
module load cmake/3.14.4

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load cuda/10.2.89 
# module load openmpi/4.0.3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/common/software/sles15_cgpu/ucx/1.9.0/lib
export PATH=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89:$PATH
mpicc=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpicc
mpicxx=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpicxx
mpif90=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpif90
mpirun=/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/bin/mpirun



export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

CCC=$mpicc
CCCPP=$mpicxx
FTN=$mpif90

#pip uninstall -r requirements.txt
#env CC=$CCC pip install --upgrade --user -r requirements.txt
env CC=$CCC pip install --user -r requirements.txt




wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=$CCC \
    -DCMAKE_Fortran_COMPILER=$FTN \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DBLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so" \
	-DLAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so"
make -j8  
cd ../../
export SCALAPACK_LIB="$PWD/scalapack-2.1.0/build/lib/libscalapack.so" 



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
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make
cp lib_gptuneclcm.so ../.
cp pdqrdriver ../



cd ../examples/
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$CCC cxx=$CCCPP prefix=$PWD/install
make install > make_parmetis_install.log 2>&1

cd ../ 
cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a $PWD/parmetis-4.0.3/install/lib/.
cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.
PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
PARMETIS_LIBRARIES="$PWD/parmetis-4.0.3/install/lib/libparmetis.so;${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcudart.so"
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -I${CUDA_ROOT}/include" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_def.so;${MKLROOT}/lib/intel64/libmkl_avx.so;${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcudart.so" \
	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn
make pzdrive_spawn


cd ../../
rm -rf hypre
git clone https://github.com/hypre-space/hypre.git
cd hypre/src/
# ./configure CC=$CCC CXX=$CCCPP FC=$FTN CFLAGS="-DTIMERUSEMPI -g -O0 -v -Q"
./configure CC=$CCC CXX=$CCCPP FC=$FTN CFLAGS="-DTIMERUSEMPI"
make
cp ../../hypre-driver/src/ij.c ./test/.
make test





cd ../../
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
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
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
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.so;$PWD/../arpack-ng/build/lib/libparpack.so"
make
make install




cd ../../
rm -rf STRUMPACK
git clone https://github.com/pghysels/STRUMPACK.git
cd STRUMPACK
#git checkout 959ff1115438e7fcd96b029310ed1a23375a5bf6  # head commit has compiler error, requiring fixes
cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/. 
mkdir build
cd build

export METIS_DIR=$PWD/../../superlu_dist/parmetis-4.0.3/install
export ParMETIS_DIR=$PWD/../../superlu_dist/parmetis-4.0.3/install
export ButterflyPACK_DIR=$PWD/../../ButterflyPACK/build/lib/cmake/ButterflyPACK

cmake ../ \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=../install \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DSTRUMPACK_COUNT_FLOPS=ON \
	-DSTRUMPACK_TASK_TIMERS=ON \
	-DSTRUMPACK_USE_CUDA=ON \
	-DTPL_CUBLAS_LIBRARIES="${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcudart.so" \
	-DTPL_CUBLAS_INCLUDE_DIRS="${CUDA_ROOT}/include" \
	-DCMAKE_CUDA_FLAGS="-I/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/include" \
	-DTPL_ENABLE_SCOTCH=OFF \
	-DTPL_ENABLE_ZFP=OFF \
	-DTPL_ENABLE_PTSCOTCH=OFF \
	-DTPL_ENABLE_PARMETIS=ON \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

make install -j4
make examples -j4




# make CC=$CCC
cd ../../../
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$CCC -shared"
python setup.py install --user
# env CC=mpicc pip install --user -e .								  



cd ../
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
python setup.py build 
python setup.py install --user
# env CC=mpicc pip install --user -e .								  



cd ../
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
env CC=$CCC pip install --user -e .


cp ../patches/opentuner/manipulator.py  ~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages/opentuner/search/.
