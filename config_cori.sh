module load python3/3.7-anaconda-2019.07
module unload cray-mpich/7.7.6

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64

module use /global/common/software/m3169/cori/modulefiles
module unload openmpi
module load openmpi/4.0.1

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/


CCC=mpicc

#pip uninstall -r requirements.txt
env CC=$CCC pip install --upgrade --user -r requirements.txt
make CC=$CCC

rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$CCC -shared"
python setup.py install --user
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
mpirun -n 1  python ./demo.py
