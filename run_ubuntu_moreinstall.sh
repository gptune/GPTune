#!/bin/bash

export GPTUNEROOT=$PWD
export PATH=$GPTUNEROOT/env/bin/:$PATH
# export BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
# export LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/examples/hypre-driver/
export PYTHONWARNINGS=ignore
export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export MPIRUN="$GPTUNEROOT/openmpi-4.0.1/bin/mpirun"
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
# export SCALAPACK_LIB=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/libscalapack.so
export LD_LIBRARY_PATH=$GPTUNEROOT/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  
RUN=$MPIRUN

export OMPI_MCA_btl="^vader"  # disable vader, this causes runtime error when run in docker

for ex in test
# for ex in test Fig.2 Fig.3 Fig.3_exp Fig.4 Fig.4_exp Fig.5 Fig.5_exp Fig.6 Fig.6_exp Fig.7 Fig.7_exp Tab.4_exp 
do
if [ $ex = 'test' ];then
    cd examples
    $RUN --allow-run-as-root --oversubscribe -n 1 python ./demo.py 
    $RUN --allow-run-as-root --oversubscribe -n 1 python ./scalapack_MLA_loaddata.py -mmax 1300 -nmax 1300 -nodes 1 -cores 4 -ntask 2 -nrun 10 -machine mymachine -jobid 0
    $RUN --allow-run-as-root --oversubscribe -n 1 python ./superlu_single.py  -nodes 1 -cores 4 -ntask 1 -nrun 20 -machine mymachine

elif [ $ex = 'Fig.2' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    python ./plot_obj_demo.py
elif [ $ex = 'Fig.3' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    matlab -r parallel_model_search_cori
elif [ $ex = 'Fig.3_exp' ];then
# this example performs one MLA iteration using $\epsilon=80$ and $\delta=20$ samples of the analytical function (see Table 2). Suppose your machine has 1 node with 16 cores, run the following two configurations and compare 'time_search':xxx and 'time_model':xxx from the runlogs.    
    cd $GPTUNEROOT/examples
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./demo_parallelperformance.py -ntask 20 -nrun 80  | tee a.out_seqential # this is the sequential benchmark
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./demo_parallelperformance.py -machine mymachine -nodes 1 -cores 16 -ntask 20 -nrun 80 -distparallel 1 | tee a.out_parallel # this is parallel modeling and search
elif [ $ex = 'Fig.4' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    bash parse_plot.sh
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    bash parse_plot_perfmodel.sh
elif [ $ex = 'Fig.4_exp' ];then
# this example autotunes the analytical function using $\epsilon=20$ and $\delta=2$ with and without a performance model. Suppose your machine has 1 node with 4 cores, run the following two configuratoins and check the difference in "Oopt" and in the plots. You will notice a better optimum is found by using the performance model.  
    cd $GPTUNEROOT/examples
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./demo_perf_model.py -nrun 20 -nodes 1 -cores 4 -ntask 2 -perfmodel 0 -plot 1 | tee a.out_demo_perf0 # without a performance model, you will see "Popt  [0.5386916457874029] Oopt  0.5365493976919858" for the task "t:0.000000" and "Popt  [0.466133977547591] Oopt  0.7810616750180267" for the task "t:0.500000"
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./demo_perf_model.py -nrun 20 -nodes 1 -cores 4 -ntask 2 -perfmodel 1 -plot 1 | tee a.out_demo_perf1 # with a performance model, you will see "Popt  [0.5382320616588907] Oopt  0.536732793802327" for the task "t:0.000000" and "Popt  [0.5257108563434262] Oopt  0.5698278466330067" for the task "t:0.500000"

# This example autotunes the runtime of SCALPACK PDGEQRF on a 2000x2000 matrix with $\epsilon=10$ and $\delta=1$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations with and without a performance model, check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime. Note that for this small matrix with small core counts, the performance model may not necessarily improve the tuning performance.  
    cd $GPTUNEROOT/examples
    $MPIRUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_loaddata_modelfit.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -nprocmin_pernode 4 -ntask 1 -nrun 10 -machine mymachine -perfmodel 0 | tee a.out_qr_perf0
    $MPIRUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_loaddata_modelfit.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -nprocmin_pernode 4 -ntask 1 -nrun 10 -machine mymachine -perfmodel 1 | tee a.out_qr_perf1

elif [ $ex = 'Fig.5' ];then
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    matlab -r plot_optimum_MLAvsSingle_QR_40000_10tasks_10samples
    matlab -r plot_optimum_MLA_SYEVX

elif [ $ex = 'Fig.5_exp' ];then
# This example autotunes the runtime of SCALPACK PDGEQRF for $\delta=2$ randomly generated-sized matrices with m,n<=2000 with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations and check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime for the two matrices.
    cd $GPTUNEROOT/examples 
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./scalapack_MLA_loaddata.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine mymachine -optimization GPTune | tee a.out_qr_MLA
    
elif [ $ex = 'Fig.6' ];then
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    bash parse_plot_tunercompare.sh
    cd $GPTUNEROOT/examples/postprocess/superlu_dist/
    bash parse_plot.sh
elif [ $ex = 'Fig.6_exp' ];then
# This example autotunes the runtime of SCALPACK PDGEQRF for $\delta=2$ randomly generated-sized matrices with m,n<=2000 with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations using the three tuners by setting -optimization to 'GPTune', 'hpbandster' or 'opentuner', check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime for the two matrices. 
    cd $GPTUNEROOT/examples
    for tuner in GPTune hpbandster opentuner 
    do
        $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./scalapack_MLA_loaddata.py -mmax 2000 -nmax 2000 -nodes 1 -cores 4 -ntask 2 -nrun 20 -machine mymachine -optimization ${tuner} | tee a.out_qr_${tuner}
    done

# this example autotunes the runtime or memory of the numerical factorization of superlu_dist using a small matrix "big.rua" with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each superlu_dist run will use at most 3 MPI ranks. Run the following configurations with setting -obj to 'time' or 'memory', and -optimization to 'GPTune', 'hpbandster' or 'opentuner' 
    cd $GPTUNEROOT/examples
    for tuner in GPTune hpbandster opentuner 
    do
        $MPIRUN --allow-run-as-root --oversubscribe -n 1 python superlu_single.py -nodes 1 -cores 4 -ntask 1 -nrun 10 -obj time -optimization ${tuner} -machine mymachine | tee a.out_superlu_${tuner}
    done

elif [ $ex = 'Fig.7' ];then
    cd $GPTUNEROOT/examples/postprocess/superlu_dist/
    matlab -r plot_pareto
    matlab -r plot_pareto_MLA
elif [ $ex = 'Fig.7_exp' ];then
# this example demonstrates the multi-objective (runtime and memory) tuning of the numerical factorization of superlu_dist using three small matrices "big.rua", "g4.rua", "g20.rua" using $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each superlu_dist run will use at most 3 MPI ranks. The Pareto optima for each matrix are shown in "Popts" and "Oopts" at the bottom of the runlog.   
    cd $GPTUNEROOT/examples
    $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./superlu_MLA_MO.py  -nodes 1 -cores 4 -ntask 3 -nrun 10 -machine mymachine | tee a.out_superlu_multiobj
elif [ $ex = 'Tab.4_exp' ];then
# this example autotunes the runtime for solving the 3D Poisson equation discretized on a nx x ny x nz grid using hypre with $\epsilon=10$ samples. The grid size is randomly generated with nx,ny,nz<=40. Suppose that your machine has 1 node with 4 cores, each hypre run will use at most 3 MPI ranks. Run the following configurations with setting -optimization to 'GPTune', 'hpbandster' or 'opentuner' 
    cd $GPTUNEROOT/examples
    for tuner in GPTune hpbandster opentuner 
    do    
        $MPIRUN --allow-run-as-root --oversubscribe -n 1 python ./hypre.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 10 -nxmax 40 -nymax 40 -nzmax 40 -machine mymachine -optimization ${tuner} | tee a.out_hypre_${tuner} 
    done
fi
done