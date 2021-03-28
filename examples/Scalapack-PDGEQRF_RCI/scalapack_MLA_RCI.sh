#!/bin/bash
start=`date +%s`
# ModuleEnv='tr4-workstation-AMD1950X-openmpi-gnu'
# ModuleEnv='cori-haswell-craympich-intel'
# ModuleEnv='cori-haswell-craympich-intel'
# ModuleEnv='cori-haswell-openmpi-gnu'
# ModuleEnv='cori-haswell-openmpi-intel'
# ModuleEnv='cori-knl-openmpi-gnu'
# ModuleEnv='cori-knl-openmpi-intel'



# Get nrun and nprocmin_pernode from command line
while getopts "a:b:c:d:e:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) nprocmin_pernode=$OPTARG ;;
      c ) mmax=$OPTARG ;;
      d ) nmax=$OPTARG ;;
      e ) ntask=$OPTARG ;;
      ? ) echo "unrecognized bash option $opt" ;; # Print helpFunction in case parameter is non-existent
   esac
done


# ############### Yang's tr4 machine
# if [ $ModuleEnv = 'tr4-workstation-AMD1950X-openmpi-gnu' ]; then
#     module load gcc/9.1.0
#     module load openmpi/gcc-9.1.0/4.0.1
#     module load scalapack-netlib/gcc-9.1.0/2.0.2
#     module load python/gcc-9.1.0/3.7.4
# # fi
# ###############


# ############### Cori Haswell Openmpi+GNU
# elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
#     module load python/3.7-anaconda-2019.10
#     module unload cray-mpich
#     module swap PrgEnv-intel PrgEnv-gnu
#     export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
#     module load openmpi/4.0.1
# # fi    
# ###############

# ############### Cori Haswell Openmpi+Intel
# elif [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
#     module load python/3.7-anaconda-2019.10
#     module unload cray-mpich
#     module swap PrgEnv-gnu PrgEnv-intel 
#     module swap intel intel/19.0.3.199 
#     export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
#     module load openmpi/4.0.1
# # fi    
# ###############

# ############### Cori Haswell CrayMPICH+GNU
# elif [ $ModuleEnv = 'cori-haswell-craympich-gnu' ]; then
#     module load python/3.7-anaconda-2019.10
#     module swap PrgEnv-intel PrgEnv-gnu
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
# # fi
# ###############

# ############### Cori Haswell CrayMPICH+Intel
# elif [ $ModuleEnv = 'cori-haswell-craympich-intel' ]; then
#     module load python/3.7-anaconda-2019.10
#     module swap PrgEnv-gnu PrgEnv-intel 
#     module swap intel intel/19.0.3.199 
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
# # fi
# ###############

# ############### Cori KNL Openmpi+GNU
# elif [ $ModuleEnv = 'cori-knl-openmpi-gnu' ]; then
# 	module unload darshan
# 	module swap craype-haswell craype-mic-knl
# 	module load craype-hugepages2M
# 	module unload cray-libsci
# 	module unload cray-mpich
# 	module swap PrgEnv-intel PrgEnv-gnu
# 	module load openmpi/4.0.1
#     export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
# # fi    
# ###############


# ############### Cori KNL Openmpi+Intel
# elif [ $ModuleEnv = 'cori-knl-openmpi-intel' ]; then
# 	module unload darshan
# 	module swap craype-haswell craype-mic-knl
# 	module load craype-hugepages2M
# 	module unload cray-libsci
# 	module unload cray-mpich
# 	module swap PrgEnv-gnu PrgEnv-intel 
#     module swap intel intel/19.0.3.199 
# 	module load openmpi/4.0.1
#     export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
#     export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
# else
#     echo "Untested ModuleEnv: $ModuleEnv for RCI, please add the corresponding definitions in this file"
#     exit
# fi    

###############
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

# echo "machine: ${machine} processor: ${processor} nodes: ${nodes} cores: ${cores}"

# nrun=20                 # number of samples per task
obj=r                   # name of the objective defined in the python file
# nprocmin_pernode=1      # minimum number of mpi count per node
niter=2                 # number of repeating each application run
bunit=8                 # mb,nb is integer multiple of bunit

database="gptune.db/PDGEQRF.json"  # the phrase PDGEQRF should match the application name defined in .gptune/meta.jason
rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./scalapack_MLA_RCI.py -nrun $nrun -mmax $mmax -nmax $nmax -ntask $ntask -bunit $bunit -nprocmin_pernode $nprocmin_pernode

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
if [[ $ModuleEnv == *"openmpi"* ]]; then
############ openmpi
    echo "mpirun --allow-run-as-root -n $nproc $BINDIR/pdqrdriver $RUNDIR"
    mpirun --allow-run-as-root -n $nproc $BINDIR/pdqrdriver $RUNDIR 
else 
############ craympich
    echo "srun -n $nproc $BINDIR/pdqrdriver $RUNDIR"
    srun -n $nproc $BINDIR/pdqrdriver $RUNDIR 
fi

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

end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"

