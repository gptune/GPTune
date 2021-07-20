#git clone https://github.com/younghyunc/newtonsketch
#cp .gptune/cori.sh .gptune/meta.json
export PYTHONPATH=$PYTHONPATH:$PWD/newtonsketch
mpirun -n 1 python tune_sketch.py -nrun 100 -npilot 20
