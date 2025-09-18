#!/bin/bash

cd ..

# Provide correct path/alias/version information for your system
CC=gcc-9
CXX=g++-9
PIP="python3 -m pip"
PY=python3
PyMAJOR=3
PyMINOR=8
PyPATCH=10

export GPTUNEROOT=$PWD
export PATH=$GPTUNEROOT/env/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

cd $GPTUNEROOT
$PIP install virtualenv
rm -rf env
$PY -m venv env
source env/bin/activate

$PIP install --upgrade -r requirements.txt
cp ./patches/opentuner/manipulator.py  ./env/lib/python$PyMAJOR.$PyMINOR/site-packages/opentuner/search/.

# rm -rf scikit-optimize
# git clone https://github.com/scikit-optimize/scikit-optimize.git
# cd scikit-optimize/
# cp ../patches/scikit-optimize/space.py skopt/space/.
# $PY setup.py build
# $PY setup.py install

# cd $GPTUNEROOT
# rm -rf cGP
# git clone https://github.com/gptune/cGP
# cd cGP/
# $PY setup.py install

# cd $GPTUNEROOT
# rm -rf autotune
# git clone https://github.com/gptune/autotune.git
# cd autotune/
# # cp ../patches/autotune/problem.py autotune/.
# env CC=$CC pip install  -e .
# cd $GPTUNEROOT
# rm -rf hybridMinimization
# git clone https://github.com/gptune/hybridMinimization.git
# cd hybridMinimization/
# $PY setup.py install


if [[ $(uname -s) == "Darwin" ]]; then
    cd $GPTUNEROOT
    rm -rf pygmo2
    git clone https://github.com/esa/pygmo2.git
    cd pygmo2
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=. -DPYTHON_EXECUTABLE:FILEPATH=python -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
    make -j8
    make install
fi

cd $GPTUNEROOT
