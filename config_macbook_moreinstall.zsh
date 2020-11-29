#!/bin/zsh

rm -rf ~/.local/lib

#set up environment variables, these are also needed when running GPTune 
################################### 
export GPTUNEROOT=$PWD
export PATH=/usr/local/Cellar/python@3.7/3.7.9_2/bin/:$PATH
export PATH=$GPTUNEROOT/env/bin/:$PATH
export BLAS_LIB=/usr/local/Cellar/openblas/0.3.12_1/lib/libblas.dylib
export LAPACK_LIB=/usr/local/Cellar/lapack/3.9.0_1/lib/liblapack.dylib
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/pygmo2/build/pygmo/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/hypre-driver/
export PYTHONWARNINGS=ignore
export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export MPIRUN="$GPTUNEROOT/openmpi-4.0.1/bin/mpirun"
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPTUNEROOT/openmpi-4.0.1/lib
export LIBRARY_PATH=$LIBRARY_PATH:$GPTUNEROOT/openmpi-4.0.1/lib
export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.dylib  
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$GPTUNEROOT/scalapack-2.1.0/build/install/lib/

CC=/usr/local/Cellar/gcc/10.2.0/bin/gcc-10
FTN=/usr/local/Cellar/gcc/10.2.0/bin/gfortran-10
CPP=/usr/local/Cellar/gcc/10.2.0/bin/g++-10




# install dependencies using homebrew and virtualenv
###################################
brew install python@3.7
alias python=python3
alias pip=pip3

python -m pip install virtualenv
python -m venv env
source env/bin/activate

pip install cloudpickle
brew install tbb
brew install pagmo
brew install pybind11

brew install gcc
brew upgrade gcc   # assuming 10.2.0

brew install openblas
brew upgrade openblas  # assuming 0.3.12_1

brew install lapack
brew upgrade lapack   # assuming 3.9.0_1


# manually install dependencies from python
###################################
cd $GPTUNEROOT
python --version
pip --version
pip install --upgrade -r requirements_mac.txt
#env CC=$MPICC pip install --upgrade --user -r requirements.txt
cp patches/opentuner/manipulator.py  ./env/lib/python3.7/site-packages/opentuner/search/.


# # pip install pygmo doesn't work, build from source
cd $GPTUNEROOT
rm -rf pygmo2
git clone https://github.com/esa/pygmo2.git
cd pygmo2
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=. -DPYTHON_EXECUTABLE:FILEPATH=python
make -j8


cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install
# env CC=mpicc pip install --user -e .

cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install -e .



# # manually install dependencies from cmake and make
# ###################################

# wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
# bzip2 -d openmpi-4.0.1.tar.bz2
# tar -xvf openmpi-4.0.1.tar 
# cd openmpi-4.0.1/ 
# ./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility
# make -j8
# make install


# cd $GPTUNEROOT
# wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
# tar -xf scalapack-2.1.0.tgz
# cd scalapack-2.1.0
# rm -rf build
# mkdir -p build
# cd build
# mkdir -p install
# cmake .. \
# 	-DBUILD_SHARED_LIBS=ON \
# 	-DCMAKE_C_COMPILER=$MPICC \
# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
# 	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
# 	-DCMAKE_INSTALL_PREFIX=./install \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DBLAS_LIBRARIES="$BLAS_LIB" \
# 	-DLAPACK_LIBRARIES="$LAPACK_LIB"
# make 
# make install


# cd $GPTUNEROOT
# mkdir -p build
# cd build
# rm -rf CMakeCache.txt
# rm -rf DartConfiguration.tcl
# rm -rf CTestTestfile.cmake
# rm -rf cmake_install.cmake
# rm -rf CMakeFiles
# cmake .. \
# 	-DCMAKE_CXX_FLAGS="" \
# 	-DCMAKE_C_FLAGS="-fopenmp" \
# 	-DCMAKE_Fortran_FLAGS="-fopenmp -fallow-argument-mismatch" \
# 	-DBUILD_SHARED_LIBS=ON \
# 	-DCMAKE_CXX_COMPILER=$MPICXX \
# 	-DCMAKE_C_COMPILER=$MPICC \
# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
# 	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
# 	-DTPL_SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
# make
# cp lib_gptuneclcm.dylib ../.
# cp pdqrdriver ../



# cd $GPTUNEROOT
# cd examples/
# rm -rf superlu_dist
# git clone https://github.com/xiaoyeli/superlu_dist.git
# cd superlu_dist

# wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
# tar -xf parmetis-4.0.3.tar.gz
# cd parmetis-4.0.3/
# mkdir -p install
# make config cc=$MPICC cxx=$MPICXX prefix=$PWD/install
# make install > make_parmetis_install.log 2>&1

# cd ../
# cp $PWD/parmetis-4.0.3/build/Darwin-x86_64/libmetis/libmetis.a $PWD/parmetis-4.0.3/install/lib/.
# PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
# PARMETIS_LIBRARIES="$PWD/parmetis-4.0.3/install/lib/libparmetis.a;$PWD/parmetis-4.0.3/install/lib/libmetis.a"

# mkdir -p build
# cd build
# rm -rf CMakeCache.txt
# rm -rf DartConfiguration.tcl
# rm -rf CTestTestfile.cmake
# rm -rf cmake_install.cmake
# rm -rf CMakeFiles
# cmake .. \
# 	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRelease" \
# 	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
# 	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
# 	-DBUILD_SHARED_LIBS=OFF \
# 	-DCMAKE_CXX_COMPILER=$MPICXX \
# 	-DCMAKE_C_COMPILER=$MPICC \
# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
# 	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
# 	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
# 	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
# 	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
# make pddrive_spawn
# make pzdrive_spawn

# cd $GPTUNEROOT
# cd examples/
# rm -rf hypre
# git clone https://github.com/hypre-space/hypre.git
# cd hypre/src/
# ./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI"
# make
# cp ../../hypre-driver/src/ij.c ./test/.
# make test





