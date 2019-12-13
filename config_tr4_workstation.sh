
# export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
# export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/

CCC=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpicc
CCCPP=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpicxx
FTN=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpif90
RUN=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpirun

env CC=$CCC pip install --upgrade --user -r requirements.txt

mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-I/usr/include/python3.6m" \
	-DCMAKE_C_FLAGS="-I/usr/include/python3.6m" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so" \
	-DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so" \
	-DTPL_SCALAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/libscalapack.so"
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
PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
PARMETIS_LIBRARIES=$PWD/parmetis-4.0.3/install/lib/libparmetis.so
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
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so" \
	-DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so" \
	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn


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
env CC=$CCC pip install --user -e .
 
 
cd ../
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
env CC=$CCC pip install --user -e .
 

cd ../examples
$RUN --allow-run-as-root -n 1 python ./demo.py

cd ../examples
$RUN --allow-run-as-root -n 1 python ./scalapack.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 1 -nrun 4 -machine tr4 -jobid 0


cd ../examples
$RUN --allow-run-as-root -n 1 python ./superlu.py  -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine tr4 -jobid 0
