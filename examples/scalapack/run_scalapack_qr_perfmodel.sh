#!/bin/bash -l
cd examples 

RUN=$MPIRUN
nrun=10
mmax=1000
nmax=1000
ntask=1
nrun1=`expr $nrun / 2`
nodes=1
cores=4
nprocmin_pernode=4  # nprocmin_pernode=cores means flat MPI 
machine='mymachine'

rm -rf *.pkl
tuner='GPTune'
$RUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_loaddata_initial.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun1} -machine ${machine} -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_initial

$RUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_loaddata_modelfit.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine ${machine} -perfmodel 1 -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_modelfit

$RUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_loaddata_modelfit.py -mmax ${mmax} -nmax ${nmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine ${machine} -perfmodel 0 -jobid 0 -optimization ${tuner}| tee a.out_scalapck_ML_m${mmax}_n${nmax}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}_oversubscribe_nomodel


