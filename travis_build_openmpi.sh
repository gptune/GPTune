#!/bin/bash

touch build_mpi.log
bash log_monitor.sh build_mpi.log &
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CXX F77=$FC FC=$FC --enable-mpi1-compatibility &>>build_mpi.log
make -j8 &>>build_mpi.log
make install &>>build_mpi.log
echo "this is finished" >> build_mpi.log
