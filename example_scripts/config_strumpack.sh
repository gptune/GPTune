cd ..
export ROOT=$PWD

###################################################
# change these to your compiler and openmpi settings
export CC=gcc-8
export FTN=gfortran-8
export CPP=g++-8
export PATH=$PATH:$ROOT/openmpi-4.1.0/bin
export MPICC="$ROOT/openmpi-4.1.0/bin/mpicc"
export MPICXX="$ROOT/openmpi-4.1.0/bin/mpicxx"
export MPIF90="$ROOT/openmpi-4.1.0/bin/mpif90"
export LD_LIBRARY_PATH=$ROOT/openmpi-4.1.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROOT/openmpi-4.1.0/lib:$LIBRARY_PATH   
###################################################



export SCALAPACK_LIB=$ROOT/scalapack-2.1.0/build/install/lib/libscalapack.a
export BLAS_LIB=$ROOT/OpenBLAS/libopenblas.a
export LAPACK_LIB=$ROOT/OpenBLAS/libopenblas.a
export LD_LIBRARY_PATH=$ROOT/OpenBLAS/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
OPENMPFLAG=fopenmp
export SCOTCH_DIR=$ROOT/scotch_6.1.0/install
export ParMETIS_DIR=$ROOT/parmetis-4.0.3/install
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$ROOT/ButterflyPACK/build/lib/cmake/ButterflyPACK


###################################
cd $ROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN install -j32


cd $ROOT
rm -rf scalapack-2.1.0.tgz*
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_Fortran_FLAGS="-$OPENMPFLAG" \
    -DBLAS_LIBRARIES="${BLAS_LIB}" \
    -DLAPACK_LIBRARIES="${LAPACK_LIB}"
make -j32
make install



cd $ROOT
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
# cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
mkdir -p install
make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
make install > make_parmetis_install.log 2>&1
cd ../
cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.



cd $ROOT
rm -rf ButterflyPACK
git clone https://github.com/liuyangzhuan/ButterflyPACK.git
cd ButterflyPACK
mkdir build
cd build
cmake .. \
    -DCMAKE_Fortran_FLAGS=""\
    -DCMAKE_CXX_FLAGS="" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_INSTALL_LIBDIR=./lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
    -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
    -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make -j32
make install -j32



cd $ROOT
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