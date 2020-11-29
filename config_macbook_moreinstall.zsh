#!/bin/zsh

rm -rf ~/.local/lib


###################################

brew install python@3.7

brew install gcc
brew upgrade gcc

brew install openblas
brew upgrade openblas

brew install lapack
brew upgrade lapack


export BLAS_LIB=/usr/local/Cellar/openblas/0.3.12_1/lib/libblas.dylib
export LAPACK_LIB=/usr/local/Cellar/lapack/3.9.0_1/lib/liblapack.dylib
export PATH=$PATH:/usr/local/Cellar/python@3.7/3.7.9_2/bin/
alias python=/usr/local/Cellar/python@3.7/3.7.9_2/bin/python3
alias pip=/usr/local/Cellar/python@3.7/3.7.9_2/bin/pip3
export CC=/usr/local/Cellar/gcc/10.2.0/bin/gcc-10
export FTN=/usr/local/Cellar/gcc/10.2.0/bin/gfortran-10
export CPP=/usr/local/Cellar/gcc/10.2.0/bin/g++-10


wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
bzip2 -d openmpi-4.0.1.tar.bz2
tar -xvf openmpi-4.0.1.tar 
cd openmpi-4.0.1/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility
make -j4
make install
export MPICC="$PWD/bin/mpicc"
export MPICXX="$PWD/bin/mpicxx"
export MPIF90="$PWD/bin/mpif90"
export MPIRUN="$PWD/bin/mpirun"

export PATH=$PATH:$PWD/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib
export LIBRARY_PATH=$LIBRARY_PATH:$PWD/lib
cd ../


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
	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DBLAS_LIBRARIES="$BLAS_LIB" \
	-DLAPACK_LIBRARIES="$LAPACK_LIB"
make 
make install
export SCALAPACK_LIB="$PWD/install/lib/libscalapack.dylib"  
cd ../../






export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/hypre-driver/
export PYTHONWARNINGS=ignore

python --version
pip --version

pip install --upgrade --user -r requirements_mac.txt
#env CC=$MPICC pip install --upgrade --user -r requirements.txt


mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="" \
	-DCMAKE_C_FLAGS="-fopenmp" \
	-DCMAKE_Fortran_FLAGS="-fopenmp -fallow-argument-mismatch" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
	-DTPL_SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
make
cp lib_gptuneclcm.dylib ../.
cp pdqrdriver ../


cd ../examples/
rm -rf superlu_dist
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config cc=$MPICC cxx=$MPICXX prefix=$PWD/install
make install > make_parmetis_install.log 2>&1

cd ../
cp $PWD/parmetis-4.0.3/build/Darwin-x86_64/libmetis/libmetis.a $PWD/parmetis-4.0.3/install/lib/.
PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
PARMETIS_LIBRARIES="$PWD/parmetis-4.0.3/install/lib/libparmetis.a;$PWD/parmetis-4.0.3/install/lib/libmetis.a"



mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRelease" \
	-DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn
make pzdrive_spawn

cd ../../
rm -rf hypre
git clone https://github.com/hypre-space/hypre.git
cd hypre/src/
./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI"
make
cp ../../hypre-driver/src/ij.c ./test/.
make test

# # pip install pygmo doesn't work, build from source
# brew install pagmo
# brew install boost
# brew install boost-python3
# rm -rf pagmo2


# install pygmo from conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
zsh ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda init zsh
conda config --add channels conda-forge
conda install -y pygmo



cd ../../../
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install
# env CC=mpicc pip install --user -e .

cd ../
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install --user -e .


cp ../patches/opentuner/manipulator.py  ~/.local/lib/python3.7/site-packages/opentuner/search/.


