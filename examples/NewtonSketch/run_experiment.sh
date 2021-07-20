#git clone https://github.com/younghyunc/newtonsketch
cp .gptune/imac.json .gptune/cori.sh
mpirun -n 1 python3 tune_sketch.py -nrun 100 -npilot 20
