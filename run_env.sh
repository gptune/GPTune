#!/bin/bash


##################################################
##################################################

# ################ frontier
# export machine=frontier
# export proc=EPYC   # knl,haswell,gpu
# export mpi=craympich    # craympich
# export compiler=gnu   # gnu, intel	
# export nodes=1  # number of nodes to be used



# ################ crusher
# export machine=crusher
# export proc=EPYC   # knl,haswell,gpu
# export mpi=craympich    # craympich
# export compiler=gnu   # gnu, intel	
# export nodes=1  # number of nodes to be used


#  # ################ summit
#   export machine=summit
#   export proc=power9   
#   export mpi=spectrummpi  
#   export compiler=gnu   
#   export nodes=1  # number of nodes to be used


## # ################ Any mac os machine that has used config_macbook.zsh to build GPTune
# export machine=mac
# export proc=mchip # intel, mchip
# export mpi=openmpi
# export compiler=gnu
# export nodes=1  # number of nodes to be used


# ############### Cori
# export machine=cori
# export proc=haswell   # knl,haswell
# export mpi=openmpi  # openmpi,craympich
# export compiler=gnu   # gnu, intel	
# export nodes=16  # number of nodes to be used


# ############# Perlmutter
# export machine=perlmutter
# export proc=milan   # milan,gpu
# export mpi=openmpi #openmpi  # craympich
# export compiler=gnu   # gnu, intel
# export nodes=1  # number of nodes to be used

# # ################ Yang's tr4 machine
#  export machine=tr4-workstation
#  export proc=AMD1950X   
#  export mpi=openmpi  
#  export compiler=gnu   
#  export nodes=1  # number of nodes to be used
# # #

################ Any ubuntu/debian machine that has used config_cleanlinux.sh to build GPTune
export machine=cleanlinux
export proc=unknown
export mpi=openmpi
export compiler=gnu
export nodes=1  # number of nodes to be used


# # ################ ex3 simula
# export machine=ex3
# export proc=xeongold16q
# export mpi=openmpi
# export compiler=gnu
# export nodes=1  # number of nodes to be used



##################################################
##################################################


if [[ $NERSC_HOST = "cori" ]]; then
    # PY_VERSION=3.7
    # PY_TIME=2019.07
    # MKL_TIME=2019.3.199

    PY_VERSION=3.8
    PY_TIME=2020.11
    MKL_TIME=2020.2.254
fi  





export ModuleEnv=$machine-$proc-$mpi-$compiler
############### Yang's tr4 machine
if [ $ModuleEnv = 'tr4-workstation-AMD1950X-openmpi-gnu' ]; then    
    module purge
    module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load python/gcc-9.1.0/3.7.4


    # module swap python python/gcc-9.1.0/3.8.4
    # export PYTHONPATH=~/.local/lib/python3.8/site-packages/:$PYTHONPATH
    # shopt -s expand_aliases
    # alias python='python3.8'

    # module purge
	# module load gcc/13.1.0
    # module load openmpi/gcc-13.1.0/4.0.1
    # module load scalapack-netlib/gcc-13.1.0/2.2.0
    # module load cmake/3.19.2	
	# module load python/gcc-13.1.0/3.12.4   

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/boost_1_78_0/build/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/oneTBB/build/lib/
    export PYTHONPATH=${SITE_PACKAGE_DIR}/GPTune/:$PYTHONPATH

    export MPIRUN=mpirun 
    export MPIARG=--allow-run-as-root
    cores=16
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
# fi
###############
############### ex3 simula
elif [ $ModuleEnv = 'ex3-xeongold16q-openmpi-gnu' ]; then
        module load cmake/gcc/3.26.4
        # module load python37
        module load openblas/dynamic/0.3.7
        # module load openblas/dynamic/0.3.23
        module load openmpi/gcc/64/4.1.5
        module load jq/1.6
        module load scalapack/gcc/2.0.2
        # module load openmpi/gcc/64/4.1.4
        module load metis/gcc/5.1.0
        module load parmetis/gcc/4.0.3
        module load scotch/gcc/6.0.7
        module load slurm/20.02.7
        module list
    export OMPI_MCA_btl="^openib,self"
    export OMPI_MCA_pml="^ucx"
    ulimit -s 10240
    export PATH=$PWD/env/bin/:$PATH
    export MPIRUN=mpirun 
    export MPIARG=
    cores=32
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [7,5,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [7,5,0]}}")
# fi
###############
############### macbook
elif [ $ModuleEnv = 'mac-intel-openmpi-gnu' ] || [ $ModuleEnv = 'mac-mchip-openmpi-gnu' ]; then
    
    MPIFromSource=1 # whether openmpi was built from source when installing GPTune
    if [[ $MPIFromSource = 1 ]]; then
        export PATH=$PWD/openmpi-5.0.6/bin:$PATH
        export MPIRUN="$PWD/openmpi-5.0.6/bin/mpirun"
        export LD_LIBRARY_PATH=$PWD/openmpi-5.0.6/lib:$LD_LIBRARY_PATH
        export DYLD_LIBRARY_PATH=$PWD/openmpi-5.0.6/lib:$DYLD_LIBRARY_PATH
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
    
    export PYTHONPATH=$PWD/build/GPTune/:$PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/build/
#	export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
	export PATH=$PWD/env/bin/:$PATH

	export SCALAPACK_LIB=$PWD/scalapack-2.2.0/build/install/lib/libscalapack.dylib
	export LD_LIBRARY_PATH=$PWD/scalapack-2.2.0/build/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/scalapack-2.2.0/build/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/scalapack-2.2.0/build/install/lib/:$DYLD_LIBRARY_PATH
	
    export LD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH
    cores=8
    gpus=0
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [5,0,6]},\"scalapack\":{\"version_split\": [2,2,0]},\"gcc\":{\"version_split\": [13,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [5,0,6]},\"scalapack\":{\"version_split\": [2,2,0]},\"gcc\":{\"version_split\": [13,1,0]}}")
# fi


############### Cori Haswell CrayMPICH+GNU
elif [ $ModuleEnv = 'cori-haswell-craympich-gnu' ]; then
    module load python/$PY_VERSION-anaconda-$PY_TIME
    module swap PrgEnv-intel PrgEnv-gnu
    module swap gcc gcc/8.3.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=srun
    cores=32
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi
###############

############### Cori Haswell CrayMPICH+Intel
elif [ $ModuleEnv = 'cori-haswell-craympich-intel' ]; then
    module load python/$PY_VERSION-anaconda-$PY_TIME
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=srun
    cores=32
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")    
# fi
###############



############### Cori Haswell Openmpi+GNU
elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    
    module swap gcc gcc/8.3.0
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module load openmpi/4.1.2
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp    
    module load python/$PY_VERSION-anaconda-$PY_TIME
    export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH


    # module unload python
    # USER="$(basename $HOME)"
    # PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0
    # source /usr/common/software/python/$PY_VERSION-anaconda-$PY_TIME/etc/profile.d/conda.sh
    # conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    # export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages


    export MPIRUN=mpirun
    cores=32
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi    
###############


############### Cori GPU Openmpi+GNU
elif [ $ModuleEnv = 'cori-gpu-openmpi-gnu' ]; then
    module swap gcc gcc/8.3.0
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
	module use /global/common/software/m3169/cori/modulefiles
    module load cgpu
	module load cuda/11.1.1 
	module load openmpi/4.0.1-ucx-1.9.0-cuda-10.2.89
    module load cudnn/8.0.5
    export UCX_LOG_LEVEL=error
    export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1
    export LD_LIBRARY_PATH=/usr/common/software/sles15_cgpu/ucx/1.9.0/lib:$LD_LIBRARY_PATH
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp

    export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH


    # module unload python
    # USER="$(basename $HOME)"
    # PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0-gpu
    # source /usr/common/software/python/$PY_VERSION-anaconda-$PY_TIME/etc/profile.d/conda.sh
    # conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    # export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages


    export MPIRUN=mpirun
    cores=40 # two 20-core Intel Xeson Gold 6148
    gpus=8 # 8 V100 per node
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi    
###############


############### Cori Haswell Openmpi+Intel
elif [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
    module load python/$PY_VERSION-anaconda-$PY_TIME
    module unload cray-mpich
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    module load openmpi/4.1.2
    export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=mpirun
    cores=32
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")    
# fi    
###############


############### Cori KNL Openmpi+GNU
elif [ $ModuleEnv = 'cori-knl-openmpi-gnu' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-intel PrgEnv-gnu
	module load openmpi/4.1.2
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=mpirun
    cores=64
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")    
# fi    
###############

############### Cori KNL CrayMPICH+GNU
elif [ $ModuleEnv = 'cori-knl-craympich-gnu' ]; then
    module load python/$PY_VERSION-anaconda-$PY_TIME
    module unload darshan
    module swap craype-haswell craype-mic-knl
    module swap PrgEnv-intel PrgEnv-gnu
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=srun
    cores=64
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi
###############


############### Cori KNL Openmpi+Intel
elif [ $ModuleEnv = 'cori-knl-openmpi-intel' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
	module load openmpi/4.1.2
    export MPIRUN=mpirun
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    cores=64
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,1,2]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")       

############### Perlmutter Milan with GPU CrayMPICH+GNU
elif [ $ModuleEnv = 'perlmutter-gpu-craympich-gnu' ]; then
    PY_VERSION=3.11
    GCC_VERSION=11.2.0
    LIBSCI_VERSION=23.02.1.1
    MPICH_VERSION=8.1.25
    CUDA_VERSION=11.7
    module load python/$PY_VERSION
	module load PrgEnv-gnu
	module load gcc/${GCC_VERSION}
	module load cray-libsci/${LIBSCI_VERSION}
	module load cray-mpich/${MPICH_VERSION}	
	module load cudatoolkit/${CUDA_VERSION}

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/lib/
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/boost_1_68_0/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/common/software/nersc/pm-2021q4/spack/cray-sles15-zen3/boost-1.78.0-ixcb3d5/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/oneTBB/build/lib/
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=srun
    export MPI4PY_RC_RECV_MPROBE=false
    export MPICH_VERSION_DISPLAY=0
    export MPICH_SINGLE_HOST_ENABLED=0
    export PMI_SPAWN_SRUN_ARGS="--mpi=cray_shasta --exclusive --network=single_node_vni,job_vni,def_tles=0 -u --cpu-bind=none"
    cores=64 # 1 socket of 64-core AMD EPYC 7763 (Milan)
    gpus=4 # 4 A100 per GPU node
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [${MPICH_VERSION//./,}]},\"libsci\":{\"version_split\": [${LIBSCI_VERSION//./,}]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]},\"cuda\":{\"version_split\": [11,7]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [${MPICH_VERSION//./,}]},\"libsci\":{\"version_split\": [${LIBSCI_VERSION//./,}]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]},\"cuda\":{\"version_split\": [11,7]}}")
# fi
###############


############### Perlmutter Milan with GPU OpenMPI+GNU
elif [ $ModuleEnv = 'perlmutter-gpu-openmpi-gnu' ]; then
    PY_VERSION=3.11
    # PY_TIME=2021.11
    GCC_VERSION=11.2.0
    OPENMPI_VERSION=5.0.3
    CUDA_VERSION=11.7
    module load python/$PY_VERSION
	module use /global/common/software/m3169/perlmutter/modulefiles
	export CRAYPE_LINK_TYPE=dynamic
    module load PrgEnv-gnu
	module load gcc/${GCC_VERSION}
	module unload cray-libsci
	module unload cray-mpich
	module unload openmpi
	module load openmpi/${OPENMPI_VERSION}
	module unload darshan
	module load cudatoolkit/${CUDA_VERSION}

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/lib/
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/boost_1_68_0/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/common/software/nersc/pm-2021q4/spack/cray-sles15-zen3/boost-1.78.0-ixcb3d5/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/oneTBB/build/lib/
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    
    # # the following was the workaround for openmpi 4.1.x, however which fails after openmpi 5.0.0 is available on Perlmutter
    # export UCX_NET_DEVICES=mlx5_0:1
    # export UCX_TLS=rc
    #### the following is the workaround for openmpi 5.0.0
    export OMPI_MCA_coll=self,libnbc,basic
    
    export MPIRUN=mpirun
    cores=64 # 1 socket of 64-core AMD EPYC 7763 (Milan)
    gpus=4
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [${OPENMPI_VERSION//./,}]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [${OPENMPI_VERSION//./,}]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
# fi 

############### Perlmutter Milan with no GPU OpenMPI+GNU
elif [ $ModuleEnv = 'perlmutter-milan-openmpi-gnu' ]; then
    PY_VERSION=3.11
    # PY_TIME=2021.11
    GCC_VERSION=11.2.0
    OPENMPI_VERSION=5.0.3
    module load python/$PY_VERSION
	module use /global/common/software/m3169/perlmutter/modulefiles
	export CRAYPE_LINK_TYPE=dynamic
    module load PrgEnv-gnu
	module load gcc/${GCC_VERSION}
	module unload cray-libsci
	module unload cray-mpich
	module unload openmpi
	module load openmpi/${OPENMPI_VERSION}
	module unload darshan

    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OMPI_DIR}/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/oneTBB/build/lib/
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/boost_1_68_0/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/common/software/nersc/pm-2021q4/spack/cray-sles15-zen3/boost-1.78.0-ixcb3d5/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    # # the following was the workaround for openmpi 4.1.x, however which fails after openmpi 5.0.0 is available on Perlmutter
    # export UCX_NET_DEVICES=mlx5_0:1
    # export UCX_TLS=rc
    #### the following is the workaround for openmpi 5.0.0
    export OMPI_MCA_coll=self,libnbc,basic
    export MPIRUN=mpirun
    cores=128 # 2 sockets of 64-core AMD EPYC 7763 (Milan)
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [${OPENMPI_VERSION//./,}]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [${OPENMPI_VERSION//./,}]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
# fi 

############### Perlmutter Milan with no GPU CrayMPICH+GNU
elif [ $ModuleEnv = 'perlmutter-milan-craympich-gnu' ]; then
    PY_VERSION=3.11
    GCC_VERSION=11.2.0
    LIBSCI_VERSION=23.02.1.1
    MPICH_VERSION=8.1.25
    module load python/$PY_VERSION
	module load PrgEnv-gnu
	module load gcc/${GCC_VERSION}
	module load cray-libsci/${LIBSCI_VERSION}
	module load cray-mpich/${MPICH_VERSION}	


    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/oneTBB/build/lib/
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/boost_1_68_0/build/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/common/software/nersc/pm-2021q4/spack/cray-sles15-zen3/boost-1.78.0-ixcb3d5/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=~/.local/perlmutter/python-$PY_VERSION/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export MPIRUN=srun
    export MPI4PY_RC_RECV_MPROBE=false
    export MPICH_VERSION_DISPLAY=0
    export MPICH_SINGLE_HOST_ENABLED=0
    export PMI_SPAWN_SRUN_ARGS="--mpi=cray_shasta --exclusive --network=single_node_vni,job_vni,def_tles=0 -u --cpu-bind=none"    
    cores=128 # 2 socket2 of 64-core AMD EPYC 7763 (Milan)
    gpus=0
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [${MPICH_VERSION//./,}]},\"libsci\":{\"version_split\": [${LIBSCI_VERSION//./,}]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [${MPICH_VERSION//./,}]},\"libsci\":{\"version_split\": [${LIBSCI_VERSION//./,}]},\"gcc\":{\"version_split\": [${GCC_VERSION//./,}]}}")
# fi
###############


elif [ $ModuleEnv = 'cleanlinux-unknown-openmpi-gnu' ]; then
    export OMPI_MCA_btl="^vader"  # disable vader, this causes runtime error when run in docker
    MPIFromSource=1 # whether openmpi was built from source when installing GPTune

    if [[ $MPIFromSource = 1 ]]; then
        export PATH=$PWD/openmpi-4.1.5/bin:$PATH
        export MPIRUN=$PWD/openmpi-4.1.5/bin/mpirun
        export LD_LIBRARY_PATH=$PWD/openmpi-4.1.5/lib:$LD_LIBRARY_PATH
    else
        export PATH=$PATH
        export MPIRUN=
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH     
        if [[ -z "$MPIRUN" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi has not been built from source when installing GPTune, please set MPIRUN, PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above."
			exit
		fi       
    fi

    export PYTHONPATH=$PWD/build/GPTune/:$PYTHONPATH
	export PATH=$PWD/env/bin/:$PATH
	export LD_LIBRARY_PATH=$PWD/scalapack-2.2.0/build/install/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PWD/OpenBLAS:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/boost_1_78_0/build/lib
    
    cores=4
    gpus=0
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,1,5]},\"scalapack\":{\"version_split\": [2,2,0]},\"gcc\":{\"version_split\": [13,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,1,5]},\"scalapack\":{\"version_split\": [2,2,0]},\"gcc\":{\"version_split\": [13,1,0]}}")

############### Cori Haswell CrayMPICH+GNU
elif [ $ModuleEnv = 'summit-power9-spectrummpi-gnu' ]; then

    module load gcc/9.1.0
    module load essl
    module load netlib-lapack
    module load netlib-scalapack
    module load cmake
    module load cuda
    module load python
  #  module load boost
    PY_VERSION=3.8

    PREFIX_PATH=$PYTHONUSERBASE

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export PATH=$PATH:$PWD/jq-1.6
    export PYTHONPATH=$PYTHONPATH:$PWD/openturns/build/share/gdb/auto-load/$PWD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/boost_1_78_0/build/lib
     
    export MPIRUN=jsrun
    cores=42
    gpus=6 # 6 V100 per node
    software_json=$(echo ",\"software_configuration\":{\"spectrum-mpi\":{\"version_split\": [10,4,0]},\"netlib-scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [9,1,0]},\"essl\":{\"version_split\": [6,1,0]},\"netlib-lapack\":{\"version_split\": [3,9,1]},\"cuda\":{\"version_split\": [11,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"spectrum-mpi\":{\"version_split\": [10,4,0]},\"netlib-scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [9,1,0]},\"essl\":{\"version_split\": [6,1,0]},\"netlib-lapack\":{\"version_split\": [3,9,1]},\"cuda\":{\"version_split\": [11,0,3]}}")

# fi

############### Crusher EPYC CrayMPICH+GNU
elif [ $ModuleEnv = 'crusher-EPYC-craympich-gnu' ]; then
    PY_VERSION=3.9
    module load cray-python
    module load cmake
    module load PrgEnv-gnu

    PREFIX_PATH=~/.local/

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export PATH=$PATH:$PWD/jq-1.6
    # export PYTHONPATH=$PYTHONPATH:$PWD/openturns/build/share/gdb/auto-load/$PWD
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/
     
    export MPIRUN=srun
    cores=64
    gpus=8 # 8 Graphics Compute Dies (GCDs), or 4 AMD MI250X per node
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [8,1,16]},\"libsci\":{\"version_split\": [21,8,1]},\"gcc\":{\"version_split\": [11,2,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [8,1,16]},\"libsci\":{\"version_split\": [21,8,1]},\"gcc\":{\"version_split\": [11,2,0]}}")

############### frontier EPYC CrayMPICH+GNU
elif [ $ModuleEnv = 'frontier-EPYC-craympich-gnu' ]; then
    PY_VERSION_FULL=3.9.13.1
    PY_VERSION=3.9
    module load cray-python/$PY_VERSION_FULL
    module load cmake
    module load PrgEnv-gnu

    PREFIX_PATH=~/.local/

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages
    export PYTHONPATH=$PREFIX_PATH/lib/python$PY_VERSION/site-packages/GPTune/:$PYTHONPATH
    export PATH=$PATH:$PWD/jq-1.6
    # export PYTHONPATH=$PYTHONPATH:$PWD/openturns/build/share/gdb/auto-load/$PWD
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/
     
    export MPIRUN=srun
    cores=32
    gpus=8 # 8 Graphics Compute Dies (GCDs), or 4 AMD MI250X per node
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [8,1,23]},\"libsci\":{\"version_split\": [22,12,1]},\"gcc\":{\"version_split\": [12,2,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [8,1,23]},\"libsci\":{\"version_split\": [22,12,1]},\"gcc\":{\"version_split\": [12,2,0]}}")

# fi

else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############

if [[ -z "${GPTUNE_LITE_MODE}" ]] && [[ $ModuleEnv == *"openmpi"* ]]; then
RUN="$MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1"
fi
if [[ -z "${GPTUNE_LITE_MODE}" ]] && [[ $ModuleEnv == *"craympich"* ]]; then
RUN="$MPIRUN --network=single_node_vni,job_vni,def_tles=0 -u --exclusive -n 1"
fi


export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
# export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
# export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD


machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")

