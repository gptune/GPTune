#!/bin/bash -l

rm -rf gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 15000 1 50
./run_cori_scalapack.sh MLA ${Config} 15000 1 50
mv gptune.db ${Config}-15000-50
mv a.out.log ${Config}-15000-50/a.out.log

rm -rf gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 20000 1 50
./run_cori_scalapack.sh MLA ${Config} 20000 1 50
mv gptune.db ${Config}-20000-50
mv a.out.log ${Config}-20000-50/a.out.log

exit

echo "ASDFASDFASD"

rm -rf gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 10000 1 50
./run_cori_scalapack.sh MLA ${Config} 10000 1 50
mv gptune.db ${Config}-10000-50
mv a.out.log ${Config}-10000-50/a.out.log

rm -rf gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 8000 1 50
./run_cori_scalapack.sh MLA ${Config} 8000 1 50
mv gptune.db ${Config}-8000-50
mv a.out.log ${Config}-8000-50/a.out.log

rm -rf gptune.db
cp -r cori-haswell-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_task ${Config} 10000 2 10
./run_cori_scalapack.sh TLA_task ${Config} 10000 2 10
mv gptune.db ${Config}-8000-10-TLA_task
mv a.out.log ${Config}-8000-10-TLA_task/a.out.log

rm -rf gptune.db
cp -r cori-haswell-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_task ${Config} 10000 2 20
./run_cori_scalapack.sh TLA_task ${Config} 10000 2 20
mv gptune.db ${Config}-8000-20-TLA_task
mv a.out.log ${Config}-8000-20-TLA_task/a.out.log

rm -rf gptune.db
cp -r cori-haswell-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_task ${Config} 10000 2 50
./run_cori_scalapack.sh TLA_task ${Config} 10000 2 50
mv gptune.db ${Config}-8000-50-TLA_task
mv a.out.log ${Config}-8000-50-TLA_task/a.out.log

##

#rm -rf gptune.db
#Config=cori-knl-openmpi-gnu-1nodes
##sbatch -W -C knl -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 10000 1 50
#./run_cori_scalapack.sh MLA ${Config} 10000 1 50
#mv gptune.db ${Config}-10000-50
#mv a.out.log ${Config}-10000-50/a.out.log

rm -rf gptune.db
cp -r cori-knl-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 10
./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 10
mv gptune.db ${Config}-10000-10-TLA_machine
mv a.out.log ${Config}-10000-10-TLA_machine/a.out.log

rm -rf gptune.db
cp -r cori-knl-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 20
./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 20
mv gptune.db ${Config}-10000-20-TLA_machine
mv a.out.log ${Config}-10000-20-TLA_machine/a.out.log

rm -rf gptune.db
cp -r cori-knl-openmpi-gnu-1nodes-10000-50 gptune.db
Config=cori-haswell-openmpi-gnu-1nodes
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 50
./run_cori_scalapack.sh TLA_machine ${Config} 10000 1 50
mv gptune.db ${Config}-10000-50-TLA_machine
mv a.out.log ${Config}-10000-50-TLA_machine/a.out.log






#rm -rf gptune.db
#Config='cori-haswell-openmpi-gnu-1nodes'
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 5000 1 50
#mv gptune.db ${Config}
#mv a.out_scalapack_MLA_${Config}.log ${Config}/a.out.log
#
#Config='cori-haswell-openmpi-gnu-1nodes'
#sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh MLA ${Config} 5000 1 50
#mv gptune.db ${Config}
#mv a.out_scalapack_MLA_${Config}.log ${Config}/a.out.log

