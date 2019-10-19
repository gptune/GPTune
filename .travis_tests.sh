#!/bin/sh
set -e

export RED="\033[31;1m"
export BLUE="\033[34;1m"
printf "${BLUE} GC; Entered tests file:\n"
export PYTHONPATH="$PYTHONPATH:$TRAVIS_BUILD_DIR/autotune/"
export PYTHONPATH="$PYTHONPATH:$TRAVIS_BUILD_DIR/scikit-optimize/"
export PYTHONPATH="$PYTHONPATH:$TRAVIS_BUILD_DIR/mpi4py/"
export MPIRUN="$TRAVIS_BUILD_DIR/installDir/openmpi-4.0.2/bin/mpirun"

export PATH="$PATH:$TRAVIS_BUILD_DIR/installDir/openmpi-4.0.2/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TRAVIS_BUILD_DIR/installDir/openmpi-4.0.2/lib/" 

case "${TEST_NUMBER}" in
1) cd $TRAVIS_BUILD_DIR
   $MPIRUN --allow-run-as-root -n 1 python ./demo.py ;;# test the demo 
*) printf "${RED} ###GC: Unknown test\n" ;;
esac

