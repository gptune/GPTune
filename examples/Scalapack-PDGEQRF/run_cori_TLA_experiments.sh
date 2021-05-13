#!/bin/bash -l

mkdir -p TLA_experiments

Config=cori-haswell-openmpi-gnu-1nodes

for tvalue in {2000,4000,6000,8000}
do
    for nrun in {10,20}
    do
        for transfer_task in {2000,4000,6000,8000}
        do
            if [[ ${tvalue} != ${transfer_task} ]]; then
                rm -rf gptune.db

                cp -r TLA_experiments/SLA-GPTune-${Config}-${transfer_task}-50 gptune.db
                #sbatch -W -C haswell -N 1 -q debug -t 00:30:00 ./run_cori_scalapack.sh TLA_task ${Config} 10000 2 20
                ./run_cori_scalapack.sh TLA_task ${Config} 10000 2 ${nrun} 1 GPTune ${tvalue} ${transfer_task}
                mkdir TLA_experiments/TLA-${Config}-${tvalue}-${transfer_task}-${nrun}
                mv gptune.db TLA_experiments/TLA-${Config}-${tvalue}-${transfer_task}-${nrun}
                mv a.out.log TLA_experiments/TLA-${Config}-${tvalue}-${transfer_task}-${nrun}/a.out.log
                exit
            fi
        done
    done
done

#for optimization in {"opentuner","hpbandster"}
for optimization in {"GPTune","opentuner","hpbandster"}
do
    for tvalue in {2000,4000,6000,8000}
    do
        for nrun in {10,20,50}
        do
            rm -rf gptune.db

            #sbatch -W -C haswell -N 64 -q debug -t 00:30:00
            ./run_cori_scalapack.sh MLA ${Config} ${tvalue} 1 ${nrun} 1 ${optimization}
            if [[ ${optimization} = 'GPTune' ]]; then
                mv gptune.db TLA_experiments/SLA-${optimization}-${Config}-${tvalue}-${nrun}
            fi
            mv a.out.log TLA_experiments/SLA-${optimization}-${Config}-${tvalue}-${nrun}/a.out.log
        done
    done
done

exit


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

