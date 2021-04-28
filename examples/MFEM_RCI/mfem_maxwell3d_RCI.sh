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


database="gptune.db/MFEM.json"  # the phrase SuperLU_DIST should match the application name defined in .gptune/meta.jason
# rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./mfem_maxwell3d_RCI.py -nprocmin_pernode $nprocmin_pernode -nrun $nrun -obj $obj

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
elif [[ $ModuleEnv == *"craympich"* ]]; then
############ craympich
    echo "srun -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help"
    srun -n $nproc $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help | tee ${logfile}
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
    
    echo "jsrun -b packed:$OMP_NUM_THREADS -d packed --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host $RS_PER_HOST '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help | tee ${logfile}"
    jsrun -b packed:$OMP_NUM_THREADS -d packed --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host $RS_PER_HOST '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' $RUNDIR/ex3p_indef -m $INPUTDIR/$mesh.mesh -x $extra -sp --omega $omega --sp_reordering_method ${sp_reordering_method} --sp_compression $sp_compression --hodlr_butterfly_levels $hodlr_butterfly_levels --sp_print_root_front_stats --sp_maxit 1000 --sp_enable_METIS_NodeNDP --sp_hodlr_min_sep_size ${sp_hodlr_min_sep_size} --sp_compression_min_front_size ${sp_compression_min_front_size} --sp_blr_min_sep_size ${sp_blr_min_sep_size} --blr_leaf_size $blr_leaf_size --${n15} --hodlr_leaf_size ${hodlr_leaf_size} --hodlr_rel_tol $tol --blr_rel_tol $tol --hodlr_max_rank 1000 --hodlr_BACA_block_size $blocksize --hodlr_verbose --help | tee ${logfile}
    
fi


iter=$(grep 'number of Krylov iterations =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')

if [[ $iter == 1000 ]]; then
    result=$bigval
else 
    # get the result (for this example: search the runlog)
    time=$(grep 'MFEM solve time =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    mem=$(grep 'factor memory =' $logfile | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    if [ $obj = time ]
    then
    result=$time
    else
    result=$mem
    fi
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

