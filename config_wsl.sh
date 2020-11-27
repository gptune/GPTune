  sudo apt-get update
  sudo apt-get install -y python
  sudo apt-get install -y git 
  sudo apt-get install -y make 
  sudo apt-get install -y cmake 
  sudo apt-get install -y zlib1g-dev
  sudo apt-get install -y libffi-dev
  sudo apt-get install -y libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.5 libgdm-dev libdb4o-cil-dev libpcap-dev
  sudo apt-get install -y libsm6 libxext6 libxrender-dev
  
  export GPROOT=$PWD
  export BLUE="\033[34;1m"
  mkdir -p installDir
  cd $GPROOT/installDir

  
  printf  "${BLUE} GC; Installing gcc-7 via apt \n"  
  sudo apt install -y gcc-7
  sudo apt install -y g++-7
  sudo apt install -y gfortran-7
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
  sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 60  
  export CXX="g++"
  export CC="gcc"
  export FC="gfortran"
  printf "${BLUE} GC; Done installing gcc-7 via apt\n"

  
  # # # printf  "${BLUE} GC; Installing gcc-9 from source \n"
  # # # sudo apt-get -y install libgmp-dev
  # # # sudo apt-get -y install libmpc-dev
  # # # sudo apt-get -y install libmpfr-dev
  # # # mkdir -p installDir
  # # # cd $GPROOT/installDir
  # # # wget https://ftpmirror.gnu.org/gcc/gcc-9.1.0/gcc-9.1.0.tar.gz
  # # # tar -xvf gcc-9.1.0.tar.gz &>build_gcc.log
  # # # cd gcc-9.1.0/
  # # # mkdir build
  # # # cd build
  # # # cp $GPROOT/travis_build_gcc.sh .
  # # # cp $GPROOT/log_monitor.sh .
  # # # bash travis_build_gcc.sh 
  # # # make install-strip
  # # # sudo update-alternatives --install /usr/bin/gcc gcc $GPROOT/installDir/gcc-9.1.0/bin/gcc 60 --slave /usr/bin/g++ g++ $GPROOT/installDir/gcc-9.1.0/bin/g++
  # # # sudo update-alternatives --install /usr/bin/gfortran gfortran $GPROOT/installDir/gcc-9.1.0/bin/gfortran 60
  # # # export CXX="g++"
  # # # export CC="gcc"
  # # # export FC="gfortran"
  # # # export PATH=$PATH:$GPROOT/installDir/gcc-9.1.0/bin
  # # # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPROOT/installDir/gcc-9.1.0/lib64  
  # # # printf "${BLUE} GC; Done installing gcc-9 from source\n"
    
	
	
  printf "${BLUE} GC; Installing python2.7.16 with gcc-9\n"
  cd $GPROOT/installDir
  wget https://www.python.org/ftp/python/2.7.16/Python-2.7.16.tgz
  tar -xvf Python-2.7.16.tgz &>build_python2.log
  cd $GPROOT/installDir/Python-2.7.16 
  cp $GPROOT/travis_build_python2.sh .
  cp $GPROOT/log_monitor.sh .
  bash travis_build_python2.sh
    #make -j8   
    #make install
  sudo rm /opt/pyenv/shims/python*
  sudo cp ./bin/python2.7 /usr/bin/python2.7.16
  sudo cp ./bin/python2.7 /usr/bin/python2.7
  sudo cp ./bin/python2.7 /usr/bin/python
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPROOT/installDir/Python-2.7.16/lib  
  export PATH=$PATH:$GPROOT/installDir/Python-2.7.16/bin
  export PYTHONHOME=$GPROOT/installDir/Python-2.7.16/
  printf "${BLUE} GC; Done Installing python2.7.16 with gcc-9\n"  
  
  
  
  printf "${BLUE} GC; Installing python3.7.4 with gcc-9\n"
  cd $GPROOT/installDir
  wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
  tar -xvf Python-3.7.4.tgz &>build_python3.log
  cd $GPROOT/installDir/Python-3.7.4
  cp $GPROOT/travis_build_python3.sh .
  cp $GPROOT/log_monitor.sh .
  bash travis_build_python3.sh
    #make -j8
    #make altinstall
  sudo cp ./bin/python3.7 /usr/bin/python
  sudo cp ./bin/python3.7 /usr/bin/python3
  sudo cp ./bin/python3.7 /usr/bin/python3.7
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPROOT/installDir/Python-3.7.4/lib  
  export PATH=$PATH:$GPROOT/installDir/Python-3.7.4/bin
  export PYTHONHOME=$GPROOT/installDir/Python-3.7.4/
  printf "${BLUE} GC; Done Installing python3.7.4 with gcc-9\n"
  unset PYTHONHOME

  printf "${BLUE} GC; Installing bzip2 apt\n"
  sudo apt-get -y install bzip2
  printf "${BLUE} GC; Done installing bzip2 apt\n"
  
  
  printf "${BLUE} GC; Installing openmpi\n"
  cd $GPROOT/installDir
  wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.bz2
  bzip2 -d openmpi-4.0.2.tar.bz2
  tar -xvf openmpi-4.0.2.tar &> build_mpi.log
  cd openmpi-4.0.2/ 
  cp $GPROOT/travis_build_openmpi.sh .
  cp $GPROOT/log_monitor.sh .
  bash travis_build_openmpi.sh  
    #make -j8 
    #make install  
  sudo cp -r $GPROOT/installDir/openmpi-4.0.2/lib/* /usr/lib/x86_64-linux-gnu/.
  export MPICC="$GPROOT/installDir/openmpi-4.0.2/bin/mpicc"
  export MPICXX="$GPROOT/installDir/openmpi-4.0.2/bin/mpicxx"
  export MPIF90="$GPROOT/installDir/openmpi-4.0.2/bin/mpif90"
  export MPIRUN="$GPROOT/installDir/openmpi-4.0.2/bin/mpirun"
  printf "${BLUE} GC; Done installing openmpi\n"
  
  printf "${BLUE} GC; Installing BLASfrom apt\n"
  sudo apt-get -y install libblas-dev 
  export BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
  printf "${BLUE} GC; Done installing BLASfrom apt\n"

  printf "${BLUE} GC; Installing LAPACKfrom apt\n"
  sudo apt-get -y install liblapack-dev
  export LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
  printf "${BLUE} GC; Done installing LAPACKfrom apt\n"
  
  printf "${BLUE} GC; Installing ScaLAPACK from source\n"
  cd $GPROOT/installDir
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
    -DBLAS_LIBRARIES="$BLAS_LIB" \
    -DLAPACK_LIBRARIES="$LAPACK_LIB"
  make -j8  &>build_scalapack.log
  sudo cp $GPROOT/installDir/scalapack-2.1.0/build/lib/libscalapack.so /usr/lib/x86_64-linux-gnu/.
  export SCALAPACK_LIB="/usr/lib/x86_64-linux-gnu/libscalapack.so"  
  printf "${BLUE} GC; Done installing ScaLAPACK from source\n"
  


  export BLUE="\033[34;1m"
  printf "${BLUE} GC; Installing GPtune from source\n"
  cd $GPROOT
  export PYTHONPATH="$PYTHONPATH:$GPROOT/autotune/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/scikit-optimize/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/mpi4py/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/GPTune/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/examples/scalapack-driver/spt/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/installDir/Python-3.7.4/lib/python3.7/site-packages"
  export PYTHONWARNINGS=ignore

  rm -rf /usr/bin/lsb_release
  env CC=$MPICC pip3.7 install -r requirements.txt &>build_gptune.log

  mkdir -p build
  cd build
  rm -rf CMakeCache.txt
  rm -rf DartConfiguration.tcl
  rm -rf CTestTestfile.cmake
  rm -rf cmake_install.cmake
  rm -rf CMakeFiles
  cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
    -DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
    -DTPL_SCALAPACK_LIBRARIES=$SCALAPACK_LIB
  make &>>build_gptune.log
  cp lib_gptuneclcm.so ../.
  cp pdqrdriver ../



  cd ../examples/
  git clone https://github.com/xiaoyeli/superlu_dist.git
  cd superlu_dist

  wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
  tar -xf parmetis-4.0.3.tar.gz
  cd parmetis-4.0.3/
  mkdir -p install
  make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
  make install > make_parmetis_install.log 2>&1

  cd ../
  PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
  PARMETIS_LIBRARIES=$PWD/parmetis-4.0.3/install/lib/libparmetis.so
  mkdir -p build
  cd build
  cmake .. \
    -DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
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
  make pddrive_spawn &>>build_gptune.log
  make pzdrive_spawn &>>build_gptune.log

  cd ../../
  rm -rf hypre
  git clone https://github.com/hypre-space/hypre.git
  cd hypre/src/
  ./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI"
  make
  cp ../../hypre-driver/src/ij.c ./test/.
  make test
	
  cd ../../../
  rm -rf mpi4py
  git clone https://github.com/mpi4py/mpi4py.git
  cd mpi4py/
  python3.7 setup.py build --mpicc="$MPICC -shared" &>>build_gptune.log
  python3.7 setup.py install &>>build_gptune.log

  # cd ../
  # rm -rf scikit-optimize
  # git clone https://github.com/scikit-optimize/scikit-optimize.git
  # cd scikit-optimize/
  # env CC=$MPICC pip3.7 install -e .
 
  cd ../
  rm -rf autotune
  git clone https://github.com/ytopt-team/autotune.git
  cd autotune/
  env CC=$MPICC pip3.7 install -e .
  printf "${BLUE} GC; Done installing GPtune from source\n"


# script:
  # cd $GPROOT/examples
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./demo.py
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 5 -machine travis -jobid 0
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./superlu_MLA_TLA.py -nodes 1 -cores 4 -ntask 1 -nrun 4 -machine travis 
  # $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 6 -machine travis  
