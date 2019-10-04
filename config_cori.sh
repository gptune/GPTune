module load python3/3.7-anaconda-2019.07
module unload cray-mpich/7.7.6
module use /global/common/software/m3169/cori/modulefiles
module load openmpi/4.0.1

env CC=mpicc pip install --upgrade --user -r requirements.txt
make CC=mpicc


 
 git clone https://github.com/ytopt-team/autotune.git
 cd autotune/
# pip install /global/homes/l/liuyangz/Cori/my_research/GPTune/autotune/
 pip install --user -e .

 git clone https://github.com/scikit-optimize/scikit-optimize.git
 cd scikit-optimize/
# pip install /global/homes/l/liuyangz/Cori/my_research/GPTune/scikit-optimize/
 pip install --user -e .


 export PYTHONPATH=/global/homes/l/liuyangz/Cori/my_research/GPTune/autotune/
 export PYTHONPATH=$PYTHONPATH:/global/homes/l/liuyangz/Cori/my_research/GPTune/scikit-optimize/
 python ./demo.py
