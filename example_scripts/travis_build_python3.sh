#!/bin/bash

touch build_python3.log
bash log_monitor.sh build_python3.log &
./configure --prefix=$PWD CC=$CC &>>build_python3.log
make -j8 &>>build_python3.log
make altinstall &>>build_python3.log
echo "this is finished" >> build_python3.log 
