module unload cray-mpich
module swap PrgEnv-intel PrgEnv-gnu
module load openmpi/4.0.1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=~/Cori/my_software/parmetis-4.0.3_dynamic_openmpi401_gnu/install/lib/:$LD_LIBRARY_PATH
./nimset | tee a.out_n${nmpi}_nimset
nmpi=512
mpirun --mca btl self,tcp,vader -N 32 --bind-to core -n ${nmpi} ./nimrod_spawn  | tee a.out_n${nmpi}
 
#for i in {1..10} 
#do
# mpirun --mca btl self,tcp,vader -N 32 --bind-to core -n 128 ../nimdevel/build_haswell_gnu_openmpi/bin/nimrod
#mpirun --mca btl self,tcp,vader -N 32 --bind-to core -n 128 ../nimdevel_oldinterface/build_haswell_gnu_openmpi/bin/nimrod
#mpirun --mca btl self,tcp,vader -N 32 --bind-to core -n 128 ../nimdevel_oldinterface/build_haswell_gnu_openmpi/bin/nimrod
#done
