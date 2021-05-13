#!/bin/bash -l

mkdir -p TLA_experiments

Config=cori-haswell-openmpi-gnu-8nodes

for tvalue in {10000,20000,30000,40000}
do
    for nrun in {10,20}
    do
        for transfer_task in {10000,20000,30000,40000}
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
    for tvalue in {10000,20000,30000,40000}
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
