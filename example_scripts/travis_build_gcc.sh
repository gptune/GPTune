#!/bin/bash

touch build_gcc.log
bash log_monitor.sh build_gcc.log &
../configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --prefix=$PWD/../ --enable-checking=release --enable-languages=c,c++,fortran --disable-multilib &>>build_gcc.log
make -j8 &>>build_gcc.log
echo "this is finished" >> build_gcc.log 
