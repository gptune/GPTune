#!/bin/bash -l

#git clone https://github.com/younghyunc/newtonsketch
#cp .gptune/cori-haswell.json .gptune/meta.json

cd ../../
source env.sh
cd -
export PYTHONPATH=$PYTHONPATH:$PWD/newtonsketch

mpirun -n 1 python tune_rrs.py -dataset epsilon_normalized_20Kn_spread -nrun 20 -npilot 10 | tee gptune.db/rrs_epsilon_normalized_20Kn_spread.a.out
mpirun -n 1 python tune_rrs.py -dataset epsilon_normalized_100Kn_spread -nrun 20 -npilot 10 | tee gptune.db/rrs_epsilon_normalized_100Kn_spread.a.out
mpirun -n 1 python tune_gaussian.py -dataset epsilon_normalized_20Kn_spread -nrun 20 -npilot 10 | tee gptune.db/gaussian_epsilon_normalized_20Kn_spread.a.out
mpirun -n 1 python tune_gaussian.py -dataset epsilon_normalized_100Kn_spread -nrun 20 -npilot 10 | tee gptune.db/gaussian_epsilon_normalized_100Kn_spread.a.out
mpirun -n 1 python tune_less_sparse.py -dataset epsilon_normalized_20Kn_spread -nrun 200 -npilot 100 | tee gptune.db/less_sparse_epsilon_normalized_20Kn_spread.a.out
mpirun -n 1 python tune_less_sparse.py -dataset epsilon_normalized_100Kn_spread -nrun 200 -npilot 100 | tee gptune.db/less_sparse_epsilon_normalized_100Kn_spread.a.out

