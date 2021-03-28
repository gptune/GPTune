#!/bin/bash

start=`date +%s`
# ModuleEnv='tr4-workstation-AMD1950X-openmpi-gnu'
# ModuleEnv='cori-haswell-craympich-gnu'
# ModuleEnv='cori-haswell-craympich-intel'
# ModuleEnv='cori-haswell-openmpi-gnu'
# ModuleEnv='cori-haswell-openmpi-intel'
# ModuleEnv='cori-knl-openmpi-gnu'
# ModuleEnv='cori-knl-openmpi-intel'

# Get nrun, nprocmin_pernode from command line
while getopts "a:b:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) nprocmin_pernode=$OPTARG ;;
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

obj1=time    # name of the objective defined in the python file
obj2=memory    # name of the objective defined in the python file



database="gptune.db/MFEM.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason
# rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./mfem_maxwell3d_MO_RCI.py -nprocmin_pernode $nprocmin_pernode -nrun $nrun 

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
mesh=${input_para[0]}
omega=${input_para[1]}


# get the tuning parameters, the parameters should follow the sequence of definition in the python file
npernode=$((2**${tuning_para[0]}))
sp_blr_min_sep_size=$((1000*${tuning_para[1]}))
sp_hodlr_min_sep_size=$((1000*${tuning_para[2]}))
blr_leaf_size=$((2**${tuning_para[3]}))
hodlr_leaf_size=$((2**${tuning_para[4]}))


sp_compression_min_front_size=50000000
sp_compression=BLR_HODLR
n15=hodlr_enable_BF_entry_n15
blocksize=64
tol=1e-5
extra=0
sp_reordering_method=metis
hodlr_butterfly_levels=100


export OMP_NUM_THREADS=$(($cores / $npernode))
nproc=$(($nodes*$npernode))

logfile=MFEM_mesh${mesh}_omega${omega}_extra${extra}_nproc${nproc}_omp${OMP_NUM_THREADS}_${sp_compression}_sp_blr_min_sep_size${sp_blr_min_sep_size}_sp_hodlr_min_sep_size${sp_hodlr_min_sep_size}_blr_leaf_size${blr_leaf_size}_hodlr_leaf_size${hodlr_leaf_size}_tol${tol}_${n15}_blocksize${blocksize}.log


RUNDIR="../MFEM/mfem/mfem-build/examples/"
INPUTDIR="../MFEM/mfem/data/"


if [[ $ModuleEnv == *"openmpi"* ]]; then
############ openmpi
    echo "mpirun --allow-run-as-root -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help"
    mpirun --allow-run-as-root -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help | tee ${logfile}
else 
############ craympich
    echo "srun -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help"
    srun -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help | tee ${logfile}
fi

iter=$(grep 'number of Krylov iterations =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')

if [[ $iter == 1000 ]]; then
    result1=$bigval
    result2=$bigval
else 
    # get the result (for this example: search the runlog)
    result1=$(grep 'MFEM solve time =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    result2=$(grep 'factor memory =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
fi


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

