#!/bin/bash


BuildExample=0 # whether all the examples have been built



##################################################
##################################################

# # ################ Any mac os machine that has used config_macbook.sh to build GPTune
# export machine=mac
# export proc=intel   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used

# ################ Cori
# export machine=cori
# export proc=haswell   # knl,haswell
# export mpi=craympich  # openmpi,craympich
# export compiler=gnu   # gnu, intel	
# export nodes=1  # number of nodes to be used


# ################ Yang's tr4 machine
# export machine=tr4-workstation
# export proc=AMD1950X   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


################ Any ubuntu/debian machine that has used config_cleanlinux.sh to build GPTune
export machine=cleanlinux
export proc=unknown   
export mpi=openmpi  
export compiler=gnu   
export nodes=1  # number of nodes to be used


##################################################
##################################################


############### automatic machine checking
if [[ $NERSC_HOST = "cori" ]]; then
    export machine=cori
elif [[ $(uname -s) = "Darwin" ]]; then
    export machine=mac
elif [[ $(dnsdomainname) = "summit.olcf.ornl.gov" ]]; then
    export machine=summit
elif [[ $(hostname -s) = "tr4-workstation" ]]; then
    export machine=tr4-workstation
elif [[ $(cat /etc/os-release | grep "PRETTY_NAME") == *"Ubuntu"* || $(cat /etc/os-release | grep "PRETTY_NAME") == *"Debian"* ]]; then
    export machine=cleanlinux
fi    






export ModuleEnv=$machine-$proc-$mpi-$compiler
############### Yang's tr4 machine
if [ $ModuleEnv = 'tr4-workstation-AMD1950X-openmpi-gnu' ]; then
    module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load python/gcc-9.1.0/3.7.4
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    MPIRUN=mpirun
    cores=16
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
# fi
###############

############### macbook
elif [ $ModuleEnv = 'mac-intel-openmpi-gnu' ]; then
    
    MPIFromSource=1 # whether openmpi was built from source when installing GPTune

    if [[ $MPIFromSource = 1 ]]; then
        export PATH=$PWD/openmpi-4.0.1/bin:$PATH
        export MPIRUN="$PWD/openmpi-4.0.1/bin/mpirun"
        export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
    else
        export MPIRUN=
        export PATH=$PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
        export LIBRARY_PATH=$LIBRARY_PATH  
        export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH    
        if [[ -z "$MPIRUN" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi has not been built from source when installing GPTune, please set MPIRUN, PATH, LD_LIBRARY_PATH, DYLD_LIBRARY_PATH for your OpenMPI build correctly above."
			exit
		fi       
    fi    
    
    export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/build/
	export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
	export PATH=$PWD/env/bin/:$PATH
	export BLAS_LIB=/usr/local/Cellar/openblas/$openblasversion/lib/libblas.dylib
	export LAPACK_LIB=/usr/local/Cellar/lapack/$lapackversion/lib/liblapack.dylib

	export SCALAPACK_LIB=$PWD/scalapack-2.1.0/build/install/lib/libscalapack.dylib
	export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$DYLD_LIBRARY_PATH
	
    export LD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH
    cores=4
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
# fi

############### Cori Haswell Openmpi+GNU
elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-intel PrgEnv-gnu
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")

############### any ubuntu machine that has used config_cleanlinux.sh to build GPTune
elif [ $ModuleEnv = 'cleanlinux-unknown-openmpi-gnu' ]; then
    export OMPI_MCA_btl="^vader"  # disable vader, this causes runtime error when run in docker
    
    MPIFromSource=1 # whether openmpi was built from source when installing GPTune

    if [[ $MPIFromSource = 1 ]]; then
        export PATH=$PWD/openmpi-4.0.1/bin:$PATH
        export MPIRUN=$PWD/openmpi-4.0.1/bin/mpirun
        export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
    else
        export PATH=$PATH
        export MPIRUN=
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH     
        if [[ -z "$MPIRUN" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi has not been built from source when installing GPTune, please set MPIRUN, PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above."
			exit
		fi       
    fi

	export PATH=$PWD/env/bin/:$PATH
	export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PWD/OpenBLAS:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/

    cores=4
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,4,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,4,0]}}")

else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############


export GPTUNEROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPy/
export PYTHONWARNINGS=ignore


machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")



RUN=$MPIRUN

for ex in test
# for ex in test Fig.2 Fig.3 Fig.3_exp Fig.4 Fig.4_exp Fig.5 Fig.5_exp Fig.6 Fig.6_exp Fig.7 Fig.7_exp Tab.4_exp 
do
if [ $ex = 'test' ];then
    cd $GPTUNEROOT/examples/GPTune-Demo
    rm -rf gptune.db/*.json
    tp=GPTune-Demo
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./demo.py 
    
    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
    rm -rf gptune.db/*.json
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA.py -mmax 1300 -nmax 1300 -ntask 2 -nrun 10 -jobid 0
    
    if [[ $BuildExample == 1 ]]; then
        cd $GPTUNEROOT/examples/SuperLU_DIST
        rm -rf gptune.db/*.json
        tp=SuperLU_DIST
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json    
        $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_MLA.py  -ntask 1 -nrun 20
    fi
elif [ $ex = 'Fig.2' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    python ./plot_obj_demo.py
elif [ $ex = 'Fig.3' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    matlab -r parallel_model_search_cori
# the plot is $GPTUNEROOT/examples/postprocess/demo/parallel_model_search.eps    
elif [ $ex = 'Fig.3_exp' ];then
# this example performs one MLA iteration using $\epsilon=80$ and $\delta=20$ samples of the analytical function (see Table 2). Suppose your machine has 1 node with 16 cores, run the following two configurations and compare 'time_search':xxx and 'time_model':xxx from the runlogs.    
    cd $GPTUNEROOT/examples/GPTune-Demo
    rm -rf gptune.db/*.json
    tp=GPTune-Demo
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json  
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./demo_parallelperformance.py -ntask 20 -nrun 80  | tee a.out_seqential # this is the sequential benchmark
    rm -rf gptune.db/*.json
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./demo_parallelperformance.py -ntask 20 -nrun 80 -distparallel 1 | tee a.out_parallel # this is parallel modeling and search
elif [ $ex = 'Fig.4' ];then
    cd $GPTUNEROOT/examples/postprocess/demo/
    bash parse_plot.sh
#the plot is $GPTUNEROOT/examples/postprocess/demo/plots/models_demo_nodes1_ntask20.pdf        
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    bash parse_plot_perfmodel.sh
#the plot is $GPTUNEROOT/examples/postprocess/scalapack/plots/models_scalapack_mmax20000_nmax20000_nodes16_ntask5.pdf        
elif [ $ex = 'Fig.4_exp' ];then
# this example autotunes the analytical function using $\epsilon=20$ and $\delta=2$ with and without a performance model. Suppose your machine has 1 node with 4 cores, run the following two configuratoins and check the difference in "Oopt" and in the plots. You will notice a better optimum is found by using the performance model.  
    cd $GPTUNEROOT/examples/GPTune-Demo
    rm -rf gptune.db/*.json
    tp=GPTune-Demo
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json      
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./demo_perf_model.py -nrun 20 -ntask 2 -perfmodel 0 -plot 1 | tee a.out_demo_perf0 # without a performance model, you will see "Popt  [0.5386916457874029] Oopt  0.5365493976919858" for the task "t:0.000000" and "Popt  [0.466133977547591] Oopt  0.7810616750180267" for the task "t:0.500000"
    rm -rf gptune.db/*.json
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./demo_perf_model.py -nrun 20 -ntask 2 -perfmodel 1 -plot 1 | tee a.out_demo_perf1 # with a performance model, you will see "Popt  [0.5382320616588907] Oopt  0.536732793802327" for the task "t:0.000000" and "Popt  [0.5257108563434262] Oopt  0.5698278466330067" for the task "t:0.500000"

# This example autotunes the runtime of SCALPACK PDGEQRF on a 2000x2000 matrix with $\epsilon=10$ and $\delta=1$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations with and without a performance model, check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime. Note that for this small matrix with small core counts, the performance model may not necessarily improve the tuning performance.  
    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json  
    $RUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_perfmodel.py -mmax 2000 -nmax 2000 -nprocmin_pernode 2 -ntask 1 -nrun 10 -perfmodel 0 | tee a.out_qr_perf0
    rm -rf gptune.db/*.json
    $RUN --oversubscribe --allow-run-as-root -n 1  python ./scalapack_MLA_perfmodel.py -mmax 2000 -nmax 2000 -nprocmin_pernode 2 -ntask 1 -nrun 10 -perfmodel 1 | tee a.out_qr_perf1

elif [ $ex = 'Fig.5' ];then
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    matlab -r plot_optimum_MLAvsSingle_QR_40000_10tasks_10samples
# the plot is $GPTUNEROOT/examples/postprocess/scalapack/scalapack_qr_MLA_optimum_64nodes.eps        
    matlab -r plot_optimum_MLA_SYEVX
# the plot is $GPTUNEROOT/examples/postprocess/scalapack/scalapack_syevx_MLA_optimum_time.eps        

elif [ $ex = 'Fig.5_exp' ];then
# This example autotunes the runtime of SCALPACK PDGEQRF for $\delta=2$ randomly generated-sized matrices with m,n<=2000 with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations and check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime for the two matrices.
    rm -rf gptune.db/*.json
    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF 
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json  
    $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA.py -mmax 2000 -nmax 2000 -ntask 2 -nrun 20 -optimization GPTune | tee a.out_qr_MLA
    
elif [ $ex = 'Fig.6' ];then
    cd $GPTUNEROOT/examples/postprocess/scalapack/
    bash parse_plot_tunercompare.sh
# the plot is $GPTUNEROOT/examples/postprocess/scalapack/plots/tuners_scalapack_mmax40000_nmax40000_nodes64_ntask10_nrun10.pdf        
    cd $GPTUNEROOT/examples/postprocess/superlu_dist/
    bash parse_plot.sh
# the plot is $GPTUNEROOT/examples/postprocess/superlu_dist/plots/tuners_superlu_dist_nodes32_ntask7_nrun20.pdf         
elif [ $ex = 'Fig.6_exp' ];then
# This example autotunes the runtime of SCALPACK PDGEQRF for $\delta=2$ randomly generated-sized matrices with m,n<=2000 with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each PDGEQRF run will use at most 3 MPI ranks. Run the following configurations using the three tuners by setting -optimization to 'GPTune', 'hpbandster' or 'opentuner', check the "Popt  [x, x, x] Oopt  x" for best tuning parameters and runtime for the two matrices. 
    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json      
    for tuner in GPTune hpbandster opentuner 
    do
        rm -rf gptune.db/*.json
        $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./scalapack_MLA.py -mmax 2000 -nmax 2000 -ntask 2 -nrun 20 -optimization ${tuner} | tee a.out_qr_${tuner}
    done
    if [[ $BuildExample == 1 ]]; then
    # this example autotunes the runtime or memory of the numerical factorization of superlu_dist using a small matrix "big.rua" with $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each superlu_dist run will use at most 3 MPI ranks. Run the following configurations with setting -obj to 'time' or 'memory', and -optimization to 'GPTune', 'hpbandster' or 'opentuner' 
        cd $GPTUNEROOT/examples/SuperLU_DIST
        tp=SuperLU_DIST
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json      
        for tuner in GPTune hpbandster opentuner 
        do
            rm -rf gptune.db/*.json
            $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python superlu_MLA.py -ntask 1 -nrun 10 -obj time -optimization ${tuner} | tee a.out_superlu_${tuner}
        done
    fi
elif [ $ex = 'Fig.7' ];then
    cd $GPTUNEROOT/examples/postprocess/superlu_dist/
# the plot is $GPTUNEROOT/examples/postprocess/superlu_dist/pareto_superlu.eps        
    matlab -r plot_pareto
# the plot is $GPTUNEROOT/examples/postprocess/superlu_dist/pareto_superlu_MLA.eps        
    matlab -r plot_pareto_MLA
elif [ $ex = 'Fig.7_exp' ];then
# this example demonstrates the multi-objective (runtime and memory) tuning of the numerical factorization of superlu_dist using three small matrices "big.rua", "g4.rua", "g20.rua" using $\epsilon=10$. Suppose that your machine has 1 node with 4 cores, each superlu_dist run will use at most 3 MPI ranks. The Pareto optima for each matrix are shown in "Popts" and "Oopts" at the bottom of the runlog.   
    if [[ $BuildExample == 1 ]]; then
        cd $GPTUNEROOT/examples/SuperLU_DIST
        rm -rf gptune.db/*.json
        tp=SuperLU_DIST
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json          
        $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./superlu_MLA_MO.py  -ntask 3 -nrun 10 | tee a.out_superlu_multiobj
    fi
elif [ $ex = 'Tab.4_exp' ];then
# this example autotunes the runtime for solving the 3D Poisson equation discretized on a nx x ny x nz grid using hypre with $\epsilon=10$ samples. The grid size is randomly generated with nx,ny,nz<=40. Suppose that your machine has 1 node with 4 cores, each hypre run will use at most 3 MPI ranks. Run the following configurations with setting -optimization to 'GPTune', 'hpbandster' or 'opentuner' 
    if [[ $BuildExample == 1 ]]; then    
        cd $GPTUNEROOT/examples/Hypre
        for tuner in GPTune hpbandster opentuner 
        do    
            rm -rf gptune.db/*.json
            tp=Hypre
            app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
            echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json                  
            $RUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py  -nprocmin_pernode 1 -ntask 1 -nrun 10 -nxmax 40 -nymax 40 -nzmax 40 -optimization ${tuner} | tee a.out_hypre_${tuner} 
        done
    fi    
fi
done
