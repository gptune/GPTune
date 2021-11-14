#!/bin/bash


##################################################
##################################################


# ################ summit
# export machine=summit
# export proc=power9   
# export mpi=spectrummpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


# ################ Any mac os machine that has used config_macbook.zsh to build GPTune
export machine=mac
export proc=intel   
export mpi=openmpi  
export compiler=gnu   
export nodes=1  # number of nodes to be used


# ############### Cori
# export machine=cori
# export proc=haswell   # knl,haswell
# export mpi=openmpi  # openmpi,craympich
# export compiler=gnu   # gnu, intel	
# export nodes=1  # number of nodes to be used


# ################ Yang's tr4 machine
# export machine=tr4-workstation
# export proc=AMD1950X   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


# ################ Any ubuntu/debian machine that has used config_cleanlinux.sh to build GPTune
# export machine=cleanlinux
# export proc=unknown   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


##################################################
##################################################





############### automatic machine checking
if [[ $(hostname -s) = "tr4-workstation" ]]; then
    export machine=tr4-workstation
elif [[ $NERSC_HOST = "cori" ]]; then
    export machine=cori
elif [[ $(uname -s) = "Darwin" ]]; then
    export machine=mac
elif [[ $(dnsdomainname) = "summit.olcf.ornl.gov" ]]; then
    export machine=summit
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
        export DYLD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$DYLD_LIBRARY_PATH
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
#	export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
	export PATH=$PWD/env/bin/:$PATH

	export SCALAPACK_LIB=$PWD/scalapack-2.1.0/build/install/lib/libscalapack.dylib
	export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$DYLD_LIBRARY_PATH
	
    export LD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH
    cores=8
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
# fi


############### Cori Haswell CrayMPICH+GNU
elif [ $ModuleEnv = 'cori-haswell-craympich-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    module swap PrgEnv-intel PrgEnv-gnu
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi
###############

############### Cori Haswell CrayMPICH+Intel
elif [ $ModuleEnv = 'cori-haswell-craympich-intel' ]; then
    module load python/3.7-anaconda-2019.10
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")    
# fi
###############



############### Cori Haswell Openmpi+GNU
elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    
    module load gcc/8.3.0
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module load openmpi/4.0.1
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp    
    module load python/3.7-anaconda-2019.10
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages


    # module unload python
    # USER="$(basename $HOME)"
    # PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0
    # source /usr/common/software/python/3.7-anaconda-2019.10/etc/profile.d/conda.sh
    # conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    # export PYTHONPATH=$PREFIX_PATH/lib/python3.7/site-packages


    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi    
###############


############### Cori GPU Openmpi+GNU
elif [ $ModuleEnv = 'cori-gpu-openmpi-gnu' ]; then
    module load gcc/8.3.0
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

    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages


    # module unload python
    # USER="$(basename $HOME)"
    # PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0-gpu
    # source /usr/common/software/python/3.7-anaconda-2019.10/etc/profile.d/conda.sh
    # conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    # export PYTHONPATH=$PREFIX_PATH/lib/python3.7/site-packages


    MPIRUN=mpirun
    cores=40 # two 20-core Intel Xeson Gold 6148
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi    
###############


############### Cori Haswell Openmpi+Intel
elif [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")    
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
	module load openmpi/4.0.1
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=64
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")    
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
	module load openmpi/4.0.1
    MPIRUN=mpirun
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    cores=64
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")       
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

############### Cori Haswell CrayMPICH+GNU
elif [ $ModuleEnv = 'summit-power9-spectrummpi-gnu' ]; then

    module swap xl gcc/7.4.0
    module load essl
    module load netlib-lapack
    module load netlib-scalapack
    module load cmake
    module load cuda/10.1.243
    module load python/3.7.0-anaconda3-5.3.0
    module load boost

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/summit/anaconda3/5.3.0/3.7/lib/python3.7/site-packages
    export PATH=$PATH:$PWD/jq-1.6
    export PYTHONPATH=$PYTHONPATH:$PWD/openturns/build/share/gdb/auto-load/$PWD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/pagmo2/build/
     
    MPIRUN=jsrun
    cores=44
    software_json=$(echo ",\"software_configuration\":{\"spectrum-mpi\":{\"version_split\": [10,3,1]},\"netlib-scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [7,4,0]},\"essl\":{\"version_split\": [6,1,0]},\"netlib-lapack\":{\"version_split\": [3,8,0]},\"cuda\":{\"version_split\": [10,1,243]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"spectrum-mpi\":{\"version_split\": [10,3,1]},\"netlib-scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [7,4,0]},\"essl\":{\"version_split\": [6,1,0]},\"netlib-lapack\":{\"version_split\": [3,8,0]},\"cuda\":{\"version_split\": [10,1,243]}}")

# fi


else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD


machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")

