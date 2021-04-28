#!/bin/bash
start=`date +%s`

# Get nrun, nprocmin_pernode, objecitve(memory or time) from command line
while getopts "a:b:c:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) nprocmin_pernode=$OPTARG ;;
      c ) obj=$OPTARG ;;
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


# obj=memory    # name of the objective defined in the python file


database="gptune.db/SuperLU_DIST.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason
rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./superlu_MLA_RCI.py -nrun $nrun -obj $obj


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
matrix=${input_para[0]}

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
COLPERM=${tuning_para[0]}
LOOKAHEAD=${tuning_para[1]}
npernode=${tuning_para[2]}
nprows=${tuning_para[3]}
NSUP=${tuning_para[4]}
NREL=${tuning_para[5]}


# call the application
npernode=$((2**$npernode))
export OMP_NUM_THREADS=$(($cores / $npernode))
export NREL=$NREL
export NSUP=$NSUP
nproc=$(($nodes*$npernode))
npcols=$(($nproc / $nprows))

RUNDIR="../SuperLU_DIST/superlu_dist/build/EXAMPLE"
INPUTDIR="../SuperLU_DIST/superlu_dist/EXAMPLE/"


if [[ $ModuleEnv == *"openmpi"* ]]; then
############ openmpi
    echo "mpirun --allow-run-as-root -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix"
    mpirun --allow-run-as-root -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix | tee a.out
elif [[ $ModuleEnv == *"craympich"* ]]; then
############ craympich
    echo "srun -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix | tee a.out"
    srun -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix | tee a.out
elif [[ $ModuleEnv == *"spectrummpi"* ]]; then
############ spectrummpi
    RS_PER_HOST=6
    GPU_PER_RS=0    # CPU only now

    if [[ $npernode -lt $RS_PER_HOST ]]; then
        npernode=$RS_PER_HOST
    fi
    export OMP_NUM_THREADS=$(($cores / $npernode))
    npernode_ext=$(($cores / $OMP_NUM_THREADS)) # break the constraint of power-of-2 npernode 
    RANK_PER_RS=$(($npernode_ext / $RS_PER_HOST)) 
    npernode_ext=$(($RANK_PER_RS * $RS_PER_HOST)) 
    RS_VAL=$(($nodes * $RS_PER_HOST)) 
    TH_PER_RS=`expr $OMP_NUM_THREADS \* $RANK_PER_RS`
    
    echo "jsrun -b packed:$OMP_NUM_THREADS -d packed --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host $RS_PER_HOST '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix | tee a.out"
    jsrun -b packed:$OMP_NUM_THREADS -d packed --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host $RS_PER_HOST '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix | tee a.out
fi


# get the result (for this example: search the runlog)
time=$(grep 'Factor time' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
mem=$(grep 'Total MEM' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
if [ $obj = time ]
then
result=$time
else
result=$mem
fi

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

