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


nodes=1                 # number of compute nodes
cores=16                # number of cores per compute node
nruns=40                # number of samples per task
machine=tr4             # name of your machine, this should match .gptune/meta.json
obj=r                   # name of the objective defined in the python file
nprocmin_pernode=1      # minimum number of mpi count per node
niter=2                 # number of repeating each application run
bunit=8                 # mb,nb is integer multiple of bunit

database="gptune.db/PDGEQRF.json"  # the phrase PDGEQRF should match the application name defined in .gptune/meta.jason
rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./scalapack_MLA_RCI.py -nodes $nodes -cores $cores -nruns $nruns -machine $machine -bunit $bunit -nprocmin_pernode $nprocmin_pernode

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


declare -a input_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].task_parameter' $database | jq -r '.[]'))
declare -a tuning_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].tuning_parameter' $database | jq -r '.[]'))


#############################################################################
#############################################################################
# Modify the following according to your application !!! 


# get the task input parameters, the parameters should follow the sequence of definition in the python file
m=${input_para[0]}
n=${input_para[1]}

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
mb=$((${tuning_para[0]}*$bunit))
nb=$((${tuning_para[1]}*$bunit))
npernode=${tuning_para[2]}
p=${tuning_para[3]}


# call the application
npernode=$((2**$npernode))
export OMP_NUM_THREADS=$(($cores / $npernode))
nproc=$(($nodes*$npernode))
q=$(($nproc / $p))

jobid=0
BINDIR=./scalapack-driver/bin/$machine/
RUNDIR=./scalapack-driver/exp/$machine/GPTune/$jobid/

# call the python wrapper to dump parameters to an input file
python ./scalapack-driver/spt/pdqrdriver_in_out.py -machine $machine -jobid $jobid -niter $niter -mode 'in' -m $m -n $n -nodes $nodes -cores $cores -mb $mb -nb $nb -nthreads $OMP_NUM_THREADS -nproc $nproc -p $p -q $q -npernode $npernode


# call the application, read data from the input file, dump results to an output file 
echo "mpirun --allow-run-as-root -n $nproc $BINDIR/pdqrdriver $RUNDIR"
mpirun --allow-run-as-root -n $nproc $BINDIR/pdqrdriver $RUNDIR 

# call the python wrapper to read results from the output file and print it out
python ./scalapack-driver/spt/pdqrdriver_in_out.py -machine $machine -jobid $jobid -niter $niter -mode 'out' -m $m -n $n -nodes $nodes -cores $cores -mb $mb -nb $nb -nthreads $OMP_NUM_THREADS -nproc $nproc -p $p -q $q -npernode $npernode | tee a.out
result=$(grep 'PDGEQRF time:' a.out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')

# write the data back to the database file
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $result '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )



#############################################################################
#############################################################################

done
done


