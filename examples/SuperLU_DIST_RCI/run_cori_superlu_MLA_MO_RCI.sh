#!/bin/bash

module load gcc/9.1.0
module load openmpi/gcc-9.1.0/4.0.1
module load scalapack-netlib/gcc-9.1.0/2.0.2
module load python/gcc-9.1.0/3.7.4


cd ../../

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

cd -


nodes=1
cores=16
ntask=1
nruns=4
machine=tr4
obj1=time    # name of the objective defined in the python file
obj2=memory    # name of the objective defined in the python file


database="gptune.db/SuperLU_DIST.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason
rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./superlu_MLA_MO_RCI.py  -nodes $nodes -cores $cores -ntask $ntask -nruns $nruns -machine $machine 

# check whether GPTune needs more data
idx=$( jq -r --arg v0 $obj1 '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )
if [ $idx = null ]
then
more=0
fi

# if so, call the application code
while [ ! $idx = null ]; 
do 
echo " $idx"    # idx indexes the record that has null objective function values


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

echo "mpirun --allow-run-as-root -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix"
mpirun --allow-run-as-root -n $nproc $RUNDIR/pddrive_spawn -c $npcols -r $nprows -l $LOOKAHEAD -p $COLPERM $INPUTDIR/$matrix > a.out


# get the result (for this example: search the runlog)
result1=$(grep 'Factor time' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
result2=$(grep 'Total MEM' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')

# write the data back to the database file
jq --arg v0 $obj1 --argjson v1 $idx --argjson v2 $result1 '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 $obj2 --argjson v1 $idx --argjson v2 $result2 '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj1 '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#############################################################################
#############################################################################

done
done


