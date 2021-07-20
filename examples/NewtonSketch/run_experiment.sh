git clone https://github.com/younghyunc/newtonsketch
cp .gptune/cori-haswell.json .gptune/cori.sh
$MPIRUN -n 1 python tune_sketch.py -nrun 100 -npilot 20
