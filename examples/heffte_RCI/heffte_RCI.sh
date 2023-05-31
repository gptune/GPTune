#!/bin/bash
start=`date +%s`

# Get nrun and npernode from command line
while getopts "a:b:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) npernode=$OPTARG ;;
      ? ) echo "unrecognized bash option $opt" ;; # Print helpFunction in case parameter is non-existent
   esac
done

cd ../../
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONWARNINGS=ignore

cd -

# name of your machine, processor model, number of compute nodes, number of cores per compute node, which are defined in .gptune/meta.json
declare -a machine_info=($(python -c "from gptune import *;
(machine, processor, nodes, cores)=list(GetMachineConfiguration());
print(machine, processor, nodes, cores)"))
machine=${machine_info[0]}
processor=${machine_info[1]}
nodes=${machine_info[2]}
cores=${machine_info[3]}

obj=time



database="gptune.db/heFFTe.json"  # the phrase heFFTe should match the application name defined in .gptune/meta.json
# rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./heffte_RCI.py -nrun $nrun -npernode $npernode


# check whether GPTune needs more data
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )
if [ $idx = null ]
then
more=0
fi

# if so, call the application code
while [ ! $idx = null ]; 
do 
echo " $idx"    # idx indexes the record that has null objective function values
# write a large value to the database. This becomes useful in case the application crashes. 
bigval=1e30
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $bigval '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database

declare -a input_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].task_parameter' $database | jq -r '.[]'))
declare -a tuning_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].tuning_parameter' $database | jq -r '.[]'))



#############################################################################
#############################################################################
# Modify the following according to your application !!! 


# get the task input parameters, the parameters should follow the sequence of definition in the python file
dimx=${input_para[0]}
dimy=${input_para[1]}
dimz=${input_para[2]}

# get the tuning parameters, the parameters should follow the sequence of definition in the python file

# note that px_i, py_i, px_o, py_o are converted from log scale to linear scale
px_i=$((2**${tuning_para[0]}))
py_i=$((2**${tuning_para[1]}))
px_o=$((2**${tuning_para[2]}))
py_o=$((2**${tuning_para[3]}))
comm_type=${tuning_para[4]}



# call the application
export OMP_NUM_THREADS=$(($cores / $npernode))


nproc=$(($nodes*$npernode))
pz_i=$(($nproc / $px_i / $py_i))
nproc=$(($px_i*$py_i*$pz_i)) # truncate the number of MPIs if not all cores are used
pz_o=$(($nproc / $px_o / $py_o))

RUNDIR="../heffte_RCI/heffte/build/benchmarks"


if [[ $ModuleEnv == *"ex3"* ]]; then
############ ex3 mpirun doesn't work correctly
    echo "srun -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out"
    srun -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out
elif [[ $ModuleEnv == *"openmpi"* ]]; then
############ openmpi
    echo "mpirun --allow-run-as-root -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out"
    mpirun --allow-run-as-root -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out
elif [[ $ModuleEnv == *"mpich"* ]]; then
############ mpich
    echo "srun -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out"
    srun -n $nproc $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out
elif [[ $ModuleEnv == *"spectrummpi"* ]]; then
############ spectrummpi
    # RS_PER_HOST=6
    # GPU_PER_RS=0    # CPU only now

    # if [[ $npernode -lt $RS_PER_HOST ]]; then
    #     npernode=$RS_PER_HOST
    # fi
    # export OMP_NUM_THREADS=$(($cores / $npernode))
    # npernode_ext=$(($cores / $OMP_NUM_THREADS)) # break the constraint of power-of-2 npernode 
    # RANK_PER_RS=$(($npernode_ext / $RS_PER_HOST)) 
    # npernode_ext=$(($RANK_PER_RS * $RS_PER_HOST)) 
    # RS_VAL=$(($nodes * $RS_PER_HOST)) 
    # TH_PER_RS=`expr $OMP_NUM_THREADS \* $RANK_PER_RS`
    
    echo "jsrun -n $nproc -a 1 -c 1   $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out"
    jsrun -n $nproc -a 1 -c 1  $RUNDIR/speed3d_c2c stock double $dimx $dimy $dimz -ingrid $px_i $py_i $pz_i -outgrid $px_o $py_o $pz_o -$comm_type | tee a.out
fi


# get the result (for this example: search the runlog)
result=$(grep 'Time per run' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')

# write the data back to the database file
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $result '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#############################################################################
#############################################################################



done
done

end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"

