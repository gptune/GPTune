#!/bin/bash -l

#git clone https://github.com/younghyunc/newtonsketch
#cp .gptune/cori-haswell.json .gptune/meta.json

cd ../../
source env.sh
cd -
export PYTHONPATH=$PYTHONPATH:$PWD/newtonsketch

mpirun -n 1 python tune_sketch.py -dataset epsilon_normalized_20Kn_0 -nrun 100 -npilot 50 | tee gptune.db/epsilon_normalized_20Kn_0.a.out
mv gptune.db/newtonsketch.json gptune.db/less_epsilon_normalized_20Kn_0.json
mpirun -n 1 python tune_sketch.py -dataset epsilon_normalized_20Kn_1 -nrun 100 -npilot 50 | tee gptune.db/epsilon_normalized_20Kn_1.a.out
mv gptune.db/newtonsketch.json gptune.db/less_epsilon_normalized_20Kn_1.json
#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn_0 -nrun 100 -npilot 50 | tee gptune.db/susy_100Kn_0.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_susy_100Kn_0.json
#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn_1 -nrun 100 -npilot 50 | tee gptune.db/susy_100Kn_1.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_susy_100Kn_1.json
#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn_2 -nrun 100 -npilot 50 | tee gptune.db/susy_100Kn_2.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_susy_100Kn_2.json
#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn_3 -nrun 100 -npilot 50 | tee gptune.db/susy_100Kn_3.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_susy_100Kn_3.json
#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn_4 -nrun 100 -npilot 50 | tee gptune.db/susy_100Kn_4.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_susy_100Kn_4.json

#mpirun -n 1 python tune_sketch.py -dataset synthetic_high_coherence -nrun 100 -npilot 50 | tee gptune.db/synthetic_high_coherence.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_synthetic_high_coherence.json
#mpirun -n 1 python tune_sketch.py -dataset cifar-10 -nrun 100 -npilot 50 | tee gptune.db/cifar-10.a.out
#mv gptune.db/newtonsketch.json gptune.db/less_cifar-10.json
#mpirun -n 1 python3 tune_sketch.py -dataset cifar-10 -nrun 40 -npilot 20 | tee cifar-10.a.out
