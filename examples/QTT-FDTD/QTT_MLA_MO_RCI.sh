#!/bin/bash
start=`date +%s`

# Get nrun and nprocmin_pernode from command line
while getopts "a:b:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) nprocmin_pernode=$OPTARG ;;
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
declare -a machine_info=($(python -c "from GPTune.gptune import *;
(machine, processor, nodes, cores)=list(GetMachineConfiguration());
print(machine, processor, nodes, cores)"))
machine=${machine_info[0]}
processor=${machine_info[1]}
nodes=${machine_info[2]}
cores=${machine_info[3]}


obj1=time    # name of the objective defined in the python file
obj2=E_diff    # name of the objective defined in the python file


database="gptune.db/QTT-FDTD.json"  # the phrase QTT-FDTD should match the application name defined in .gptune/meta.jason
# rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./QTT_MLA_MO_RCI.py -nrun $nrun 

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

# write a large value to the database. This becomes useful in case the application crashes. 
bigval=1e30
jq --arg v0 $obj1 --argjson v1 $idx --argjson v2 $bigval '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 $obj2 --argjson v1 $idx --argjson v2 $bigval '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database

declare -a input_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].task_parameter' $database | jq -r '.[]'))
declare -a tuning_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].tuning_parameter' $database | jq -r '.[]'))



#############################################################################
#############################################################################
# Modify the following according to your application !!! 


# get the task input parameters, the parameters should follow the sequence of definition in the python file
L=${input_para[0]}

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
deriv_order=${tuning_para[0]}
lgeta=${tuning_para[1]}
eta=$(echo "scale=10; e($lgeta * l(10))" | bc -l)

RUNDIR="$CFS/m2957/liuyangz/my_research/QTT-FDTD/tns_pde_v4_pr2_new/tests/EM2D"

start1=`date +%s.%N`
echo "export L=$L; export deriv_order=$deriv_order; export eta=$eta; cd $RUNDIR; bash runit_Jon.sh | tee a.out; cd -; mv $RUNDIR/a.out .
"
export L=$L; export deriv_order=$deriv_order; export eta=$eta; cd $RUNDIR; bash runit_Jon.sh | tee a.out; cd -; mv $RUNDIR/a.out .
end1=`date +%s.%N`
time_fun=$( echo "$end1 - $start1" | bc -l )
echo "time_fun: $time_fun"

# get the result (for this example: search the runlog)
result1=$(grep 'elapsed time:' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?')
result2=$(grep 'err diff Ez:' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?')

# write the data back to the database file

jq --arg v0 $obj1 --argjson v1 $idx --argjson v2 $result1 '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 $obj2 --argjson v1 $idx --argjson v2 $result2 '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj1 '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#############################################################################
#############################################################################

done
done
end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"



