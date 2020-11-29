#!/bin/zsh

rm -rf ~/.local/lib


###################################
# The following loads preinstalled compiler, openmpi, and python, make sure to change them to the correct paths on your machine
source /usr/local/Cellar/modules/4.3.0/init/zsh
module load gcc/9.2.0
module load openmpi/gcc-9.2.0/4.0.1
alias python=/usr/local/Cellar/python@3.7/3.7.9_2/bin/python3
alias pip=/usr/local/Cellar/python@3.7/3.7.9_2/bin/pip3

# set the PATH to your python installation
export PATH=$PATH:/usr/local/Cellar/python@3.7/3.7.9_2/bin

# set the compiler wrappers
CCC=$MPICC  
CCCPP=$MPICXX
FTN=$MPIF90
RUN=$MPIRUN

# set the path to blas,lapack
BLAS_LIB=/usr/local/Cellar/openblas/0.3.12_1/lib/libblas.dylib
LAPACK_LIB=/usr/local/Cellar/lapack/3.9.0_1/lib/liblapack.dylib

###################################


export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

python --version
pip --version

pip install --upgrade --user -r requirements_mac.txt
#env CC=$CCC pip install --upgrade --user -r requirements.txt


wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
mkdir -p install
cmake .. \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DBLAS_LIBRARIES="$BLAS_LIB" \
	-DLAPACK_LIBRARIES="$LAPACK_LIB"
make 
make install
export SCALAPACK_LIB="$PWD/install/lib/libscalapack.dylib"  


cd ../../
mkdir -p build
cd build
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
	-DCMAKE_Fortran_FLAGS="-fopenmp" \
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
make config cc=$CCC cxx=$CCCPP prefix=$PWD/install
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
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
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
./configure CC=$CCC CXX=$CCCPP FC=$FTN CFLAGS="-DTIMERUSEMPI"
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
python setup.py build --mpicc="$CCC -shared"
python setup.py install
# env CC=mpicc pip install --user -e .

cd ../
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
pip install --user -e .
cd ../
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install --user -e .


cp ../patches/opentuner/manipulator.py  ~/.local/lib/python3.7/site-packages/opentuner/search/.
cp ../patches/opentuner/manipulator.py  ~/Library/Python/3.7/lib/python/site-packages/opentuner/search/.


