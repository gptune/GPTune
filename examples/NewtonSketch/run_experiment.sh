#git clone https://github.com/younghyunc/newtonsketch

#cp .gptune/cori-haswell.json .gptune/meta.json

cd ../../
source env.sh
cd -
export PYTHONPATH=$PYTHONPATH:$PWD/newtonsketch

#mpirun -n 1 python tune_sketch.py -dataset susy_100Kn -nrun 40 -npilot 20 | tee susy_100Kn.a.out
mpirun -n 1 python3 tune_sketch.py -dataset synthetic_high_coherence -nrun 40 -npilot 20 | tee synthetic_high_coherence.a.out
mpirun -n 1 python3 tune_sketch.py -dataset cifar-10 -nrun 40 -npilot 20 | tee cifar-10.a.out
