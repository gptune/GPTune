#!/bin/bash
start=`date +%s`


# Get nstepmax, nstepmin, Nloop, optimization from command line
while getopts "a:b:c:d:" opt
do
   case $opt in
      a ) nstepmax=$OPTARG ;;
      b ) nstepmin=$OPTARG ;;
      c ) Nloop=$OPTARG ;;
      d ) optimization=$OPTARG ;;
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


obj=time    # name of the objective defined in the python file
bmin=1
bmax=8
eta=2

database="gptune.db/NIMROD.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason
rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./nimrod_single_MB_RCI.py -nstepmax $nstepmax -nstepmin $nstepmin -Nloop $Nloop -bmin $bmin -bmax $bmax -eta $eta -optimization $optimization


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
if [ $optimization = 'GPTune' ];then
    mx=${input_para[0]}
    my=${input_para[1]}
    lphi=${input_para[2]}
    nstep=$nstepmax
elif [ $optimization = 'GPTuneBand' ];then
    budget=${input_para[0]}
    mx=${input_para[1]}
    my=${input_para[2]}
    lphi=${input_para[3]}  

    nstep=$(python -c "b=$budget;bmin=$bmin;bmax=$bmax;nmin=$nstepmin;nmax=$nstepmax
k1 = (nmax-nmin)/(bmax-bmin)
b1 = nmin - k1
assert k1 * bmax + b1 == nmax
result = int(k1 * b + b1)
print(result)")

fi

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
elif [[ $ModuleEnv == *"craympich"* ]]; then
############ craympich
    BINDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimdevel_spawn/build_haswell_gnu_craympich/bin"
    RUNDIR="/project/projectdirs/m2957/liuyangz/my_research/nimrod/nimrod_input_craympich"
fi

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


logfile=NIMROD_mx${mx}_my${my}_lphi${lphi}_nstep${nstep}_NSUP${NSUP}_NREL${NREL}_nbx${nbx}_nby${nby}_nproc${nproc}_omp${OMP_NUM_THREADS}.log



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
result=${arr[0]}

# result=1
echo "nimrod time: mx: $mx, my: $my, lphi: $lphi, nstep: $nstep, NSUP: $NSUP, NREL: $NREL, nbx: $nbx, nby: $nby, result: $result"


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

