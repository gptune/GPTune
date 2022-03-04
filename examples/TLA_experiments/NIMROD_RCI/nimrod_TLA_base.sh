#!/bin/bash
start=`date +%s`

# Get nstep, expid, seed from command line
while getopts "a:b:c:d:e:" opt
do
   case $opt in
      a ) nstep=$OPTARG ;;
      b ) expid=$OPTARG ;;
      c ) seed=$OPTARG ;;
      d ) nrun=$OPTARG ;;
      e ) npilot=$OPTARG ;;
      ? ) echo "unrecognized bash option $opt" ;; # Print helpFunction in case parameter is non-existent
   esac
done

cd ../../../
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONWARNINGS=ignore

cd -

# number of compute nodes and number of cores per compute node
nodes=16
cores=32

obj=time    # name of the objective defined in the python file

database="gptune.db/NIMROD_TLA_base.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./nimrod_TLA_base.py -nstep $nstep -expid $expid -seed $seed -nrun ${nrun} -npilot ${npilot}


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
mx=${input_para[0]}
my=${input_para[1]}
lphi=${input_para[2]}
nstep=$nstep

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
NSUP=${tuning_para[0]}
NREL=${tuning_para[1]}
nbx=${tuning_para[2]}
nby=${tuning_para[3]}

# call the application
NTH=1
export OMP_NUM_THREADS=$NTH # flat MPI
export NREL=$NREL
export NSUP=$NSUP

nproc=$(($nodes*$cores))

if [[ $ModuleEnv == *"openmpi"* ]]; then
############ openmpi
    BINDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_haswell_gnu_openmpi/bin"
    RUNDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimrod_input"
elif [[ $ModuleEnv == *"mpich"* ]]; then
############ mpich
if [[ $ModuleEnv == *"haswell"* ]]; then
    BINDIR="/project/projectdirs/mp156/younghyun/nimrod/nimdevel_spawn/build_haswell_gnu_craympich/bin"
    RUNDIR="/project/projectdirs/mp156/younghyun/nimrod/nimrod_input_craympich"
elif [[ $ModuleEnv == *"knl"* ]]; then
    BINDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_knl_gnu_craympich/bin"
    RUNDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimrod_input_craympich"
fi
fi

mkdir nimrod_dir
cd nimrod_dir

cp $RUNDIR/nimrod.in ./nimrod_template.in
cp $RUNDIR/fluxgrid.in .
cp $RUNDIR/g163518.03130 .
cp $RUNDIR/p163518.03130 .
cp $RUNDIR/nimset .
cp $BINDIR/nimrod .


COLPERM='4'
ROWPERM='2'


tmp=$(python -c "import numpy as np
fin = open(\"./nimrod_template.in\",\"rt\")
fout = open(\"./nimrod.in\",\"wt\")
for line in fin:
    #read replace the string and write to output file
    if(line.find(\"iopts(3)\")!=-1):
        fout.write(\"iopts(3)= %s\n\"%($ROWPERM))
    elif(line.find(\"iopts(4)\")!=-1):
        fout.write(\"iopts(4)= %s\n\"%($COLPERM))    
    elif(line.find(\"lphi\")!=-1):
        fout.write(\"lphi= %s\n\"%($lphi))    
    elif(line.find(\"nlayers\")!=-1):
        fout.write(\"nlayers= %s\n\"%(int(np.floor(2**$lphi/3.0)+1)))  	
    elif(line.find(\"mx\")!=-1):
        fout.write(\"mx= %s\n\"%(2**$mx))
    elif(line.find(\"nstep\")!=-1):
        fout.write(\"nstep= %s\n\"%($nstep))  			  
    elif(line.find(\"my\")!=-1):
        fout.write(\"my= %s\n\"%(2**$my))   
    elif(line.find(\"nxbl\")!=-1):
        fout.write(\"nxbl= %s\n\"%(int(2**$mx/2**$nbx)))  
    elif(line.find(\"nybl\")!=-1):
        fout.write(\"nybl= %s\n\"%(int(2**$my/2**$nby)))  									  						        
    else:
        fout.write(line)
#close input and output files
fin.close()
fout.close()")


    nproc=$(python -c "import numpy as np    
nlayers=int(np.floor(2**$lphi/3.0)+1)
nprocmax=$nodes*$cores
nproc = int(nprocmax/nlayers)*nlayers
if(nprocmax<nlayers):
    print(0)
    raise Exception(\"nprocmax<nlayers\")
if(nproc>int(2**$mx/2**$nbx)*int(2**$my/2**$nby)*int(np.floor(2**$lphi/3.0)+1)): # nproc <= nlayers*nxbl*nybl
    nproc = int(2**$mx/2**$nbx)*int(2**$my/2**$nby)*int(np.floor(2**$lphi/3.0)+1) 
print(nproc) ")

result_arr=(0 0 0)
for repeat in {1,2,3}
do
    logfile=NIMROD_mx${mx}_my${my}_lphi${lphi}_nstep${nstep}_NSUP${NSUP}_NREL${NREL}_nbx${nbx}_nby${nby}_nproc${nproc}_omp${OMP_NUM_THREADS}_run${repeat}.log

    if [[ $ModuleEnv == *"openmpi"* ]]; then
    ############ openmpi
        echo "mpirun --mca btl self,tcp,vader --allow-run-as-root -n $nproc ./nimrod"
        ./nimset
        mpirun --mca btl self,tcp,vader -N $cores --bind-to core --allow-run-as-root -n $nproc ./nimrod | tee $logfile
    elif [[ $ModuleEnv == *"craympich"* ]]; then
    ############ craympich
        echo "srun -n $nproc ./nimrod"
        ./nimset
        THREADS_PER_RANK=`expr $NTH \* 2`
        srun -n $nproc -N $nodes -c $THREADS_PER_RANK --cpu_bind=cores ./nimrod | tee $logfile
    fi

    # get the result (for this example: search the runlog) egrep is needed for scientific notation
    declare -a arr=($(grep 'Loop  time' $logfile | egrep -o "[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?."))
    result_arr[${repeat}-1]=$(python -c "print (float(${arr[0]}))")
    #result_arr[${repeat}-1]=${arr[0]}
done

echo result1: ${result_arr[0]}
echo result2: ${result_arr[1]}
echo result3: ${result_arr[2]}

result=$(echo ${result_arr[0]}+${result_arr[1]}+${result_arr[2]} | bc -l)
result=$(echo ${result}/3 | bc -l)
echo average result: ${result}

# result=1
echo "nimrod time: mx: $mx, my: $my, lphi: $lphi, nstep: $nstep, NSUP: $NSUP, NREL: $NREL, nbx: $nbx, nby: $nby, result: $result"

cd ..

# write the data back to the database file
jq --arg v0 $obj --argjson v1 $idx --argjson v2 [${result_arr[0]},${result_arr[1]},${result_arr[2]}] '.func_eval[$v1].evaluation_detail[$v0].evaluations=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $result '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database

# update timestamp
timestamp=$(python -c "import time
now = time.localtime()
print (str(now.tm_year)+'-'+str(now.tm_mon)+'-'+str(now.tm_mday)+'-'+str(now.tm_hour)+'-'+str(now.tm_min)+'-'+str(now.tm_sec)+'-'+str(now.tm_wday)+'-'+str(now.tm_yday)+'-'+str(now.tm_isdst))")
tm_year=$(echo ${timestamp} | cut -d "-" -f 1)
tm_mon=$(echo ${timestamp} | cut -d "-" -f 2)
tm_mday=$(echo ${timestamp} | cut -d "-" -f 3)
tm_hour=$(echo ${timestamp} | cut -d "-" -f 4)
tm_min=$(echo ${timestamp} | cut -d "-" -f 5)
tm_sec=$(echo ${timestamp} | cut -d "-" -f 6)
tm_wday=$(echo ${timestamp} | cut -d "-" -f 7)
tm_yday=$(echo ${timestamp} | cut -d "-" -f 8)
tm_isdst=$(echo ${timestamp} | cut -d "-" -f 9)
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_year} '.func_eval[$v1].time.tm_year=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_mon} '.func_eval[$v1].time.tm_mon=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_mday} '.func_eval[$v1].time.tm_mday=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_hour} '.func_eval[$v1].time.tm_hour=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_min} '.func_eval[$v1].time.tm_min=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_sec} '.func_eval[$v1].time.tm_sec=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_wday} '.func_eval[$v1].time.tm_wday=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_yday} '.func_eval[$v1].time.tm_yday=$v2' $database > tmp.json && mv tmp.json $database
jq --arg v0 "time" --argjson v1 $idx --argjson v2 ${tm_isdst} '.func_eval[$v1].time.tm_isdst=$v2' $database > tmp.json && mv tmp.json $database

idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#############################################################################
#############################################################################

done
done

end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"

