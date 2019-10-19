
# export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
# export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/

CCC=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpicc
RUN=/home/administrator/Desktop/software/openmpi-4.0.2/bin/mpirun

env CC=$CCC pip install --upgrade --user -r requirements.txt
make CC=$CCC


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
 


cd ../
$RUN --allow-run-as-root -n 1 python ./demo.py
