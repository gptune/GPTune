  export GPROOT=$PWD
  export CXX="g++"
  export CC="gcc"
  export FC="gfortran"
  # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPROOT/installDir/Python-2.7.16/lib  
  # export PATH=$PATH:$GPROOT/installDir/Python-2.7.16/bin
  # export PYTHONHOME=$GPROOT/installDir/Python-2.7.16/
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPROOT/installDir/Python-3.7.4/lib  
  export PATH=$PATH:$GPROOT/installDir/Python-3.7.4/bin
  export PYTHONHOME=$GPROOT/installDir/Python-3.7.4/
  export MPICC="$GPROOT/installDir/openmpi-4.0.1/bin/mpicc"
  export MPICXX="$GPROOT/installDir/openmpi-4.0.1/bin/mpicxx"
  export MPIF90="$GPROOT/installDir/openmpi-4.0.1/bin/mpif90"
  export MPIRUN="$GPROOT/installDir/openmpi-4.0.1/bin/mpirun"
  export BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
  export LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
  export SCALAPACK_LIB="/usr/lib/x86_64-linux-gnu/libscalapack.so"  
  export PYTHONPATH="$PYTHONPATH:$GPROOT/autotune/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/scikit-optimize/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/mpi4py/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/GPTune/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/examples/scalapack-driver/spt/"
  export PYTHONPATH="$PYTHONPATH:$GPROOT/installDir/Python-3.7.4/lib/python3.7/site-packages"  
  export PYTHONWARNINGS=ignore


  cd $GPROOT/examples
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./demo.py
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_TLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_TLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 5 -machine travis -jobid 0
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./scalapack_MLA_loaddata.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine travis -jobid 0
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./superlu_MLA_TLA.py -nodes 1 -cores 4 -ntask 1 -nrun 4 -machine travis 
  $MPIRUN --allow-run-as-root --oversubscribe -n 1 python3.7 ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 1 -nrun 6 -machine travis  
