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
GPTUNEROOT=$PWD



# set the path to blas,lapack
BLAS_LIB=/usr/local/Cellar/openblas/0.3.12_1/lib/libblas.dylib
LAPACK_LIB=/usr/local/Cellar/lapack/3.9.0_1/lib/liblapack.dylib
SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.dylib  

###################################


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
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.a;$ParMETIS_DIR/lib/libmetis.a"
export METIS_LIBRARIES=$ParMETIS_DIR/lib/libmetis.a


# set the compiler wrappers
MPICC=$MPICC
MPICXX=$MPICXX
MPIF90=$MPIF90
MPIRUN=$MPIRUN




python --version
pip --version

pip install --upgrade --user -r requirements_mac.txt
#env CC=$MPICC pip install --upgrade --user -r requirements.txt


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
	-DCMAKE_INSTALL_PREFIX=./install \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DBLAS_LIBRARIES="$BLAS_LIB" \
	-DLAPACK_LIBRARIES="$LAPACK_LIB"
make 
make install


cd $GPTUNEROOT
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
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
	-DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
	-DTPL_SCALAPACK_LIBRARIES="$SCALAPACK_LIB"
make
cp lib_gptuneclcm.dylib ../.
# cp pdqrdriver ../


cd $GPTUNEROOT/examples/SuperLU_DIST
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

cd $GPTUNEROOT/examples/Hypre
rm -rf hypre
git clone https://github.com/hypre-space/hypre.git
cd hypre/src/
./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI"
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
	-DCMAKE_Fortran_FLAGS=""\
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
	-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.dylib;$PWD/../arpack-ng/build/lib/libparpack.dylib"
make -j4
make install -j4



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
sudo chmod +x examples/KernelRegressionMPI.py
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

make install -j4
make examples -j4


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
	-DSTRUMPACK_LIBRARIES="${STRUMPACK_DIR}/lib/libstrumpack.dylib;${ButterflyPACK_DIR}/../../../lib/libdbutterflypack.dylib;${ButterflyPACK_DIR}/../../../lib/libzbutterflypack.dylib;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libparpack.dylib;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libarpack.dylib;${PARMETIS_LIBRARIES};${SCALAPACK_LIB}"
make -j4 VERBOSE=1
make install
make ex3p_indef


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



cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install
# env CC=mpicc pip install --user -e .



cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
pip install --user -e .


cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
cp ../patches/autotune/problem.py autotune/.
pip install --user -e .


cp ../patches/opentuner/manipulator.py  /usr/local/lib/python3.7/site-packages/opentuner/search/.
cp ../patches/opentuner/manipulator.py  ~/Library/Python/3.7/lib/python/site-packages/opentuner/search/.


