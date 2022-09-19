#!/bin/bash


if [[ $NERSC_HOST != "cori" ]]; then
	echo "This script can only be used for Cori"
	exit
fi


# PY_VERSION=3.7
# PY_TIME=2019.07
# MKL_TIME=2019.3.199

PY_VERSION=3.8
PY_TIME=2020.11
MKL_TIME=2020.2.254


rm -rf  ~/.cache/pip
rm -rf ~/.local/cori/
rm -rf ~/.local/lib/python$PY_VERSION
module load python/$PY_VERSION-anaconda-$PY_TIME
PREFIX_PATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/

echo $(which python) 

module unload cmake
module load cmake/3.22.1


##################################################
##################################################
machine=cori
proc=haswell   # knl,haswell,gpu
mpi=openmpi    # openmpi,craympich
compiler=gnu   # gnu, intel	


BuildExample=1 # whether to build all examples

export ModuleEnv=$machine-$proc-$mpi-$compiler


##################################################
##################################################

echo "The ModuleEnv is $ModuleEnv"
if [ $ModuleEnv = 'cori-haswell-craympich-gnu' ]; then
	export CRAYPE_LINK_TYPE=dynamic
	module swap PrgEnv-intel PrgEnv-gnu
	GPTUNEROOT=$PWD
	BLAS_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=fopenmp
# fi 

elif [ $ModuleEnv = 'cori-haswell-craympich-intel' ]; then
	export CRAYPE_LINK_TYPE=dynamic
	module swap PrgEnv-gnu PrgEnv-intel 
	module swap intel intel/19.0.3.199 
	GPTUNEROOT=$PWD
	BLAS_LIB="/opt/cray/pe/libsci/20.09.1/INTEL/16.0/x86_64/lib/libsci_intel_mpi_mp.so"
	LAPACK_LIB="/opt/cray/pe/libsci/20.09.1/INTEL/16.0/x86_64/lib/libsci_intel_mpi_mp.so"
	SCALAPACK_LIB="/opt/cray/pe/libsci/20.09.1/INTEL/16.0/x86_64/lib/libsci_intel_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=qopenmp
# fi 

elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
	module swap gcc gcc/8.3.0
    module load openmpi/4.1.2
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp

	GPTUNEROOT=$PWD
	
	export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
	BLAS_INC="-I${MKLROOT}/include"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
	BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
	LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"


	######### uncomment the following to use python installed with pytorch 
	### this assumes pytorch is built from source from https://github.com/sparticlesteve/nersc-pytorch-build 
	# module unload python
	# USER="$(basename $HOME)"
	# PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0
	# source /usr/common/software/python/$PY_VERSION-anaconda-$PY_TIME/etc/profile.d/conda.sh
	# conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"	
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
	# BLAS_LIB="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;-lgomp"
	# LAPACK_LIB="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;-lgomp"
	
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"

	
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
# fi 


elif [ $ModuleEnv = 'cori-gpu-openmpi-gnu' ]; then
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
	module swap gcc gcc/8.3.0


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

	GPTUNEROOT=$PWD

	
	export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
	BLAS_INC="-I${MKLROOT}/include"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
	BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
	LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"


	# ######### uncomment the following to use python installed with pytorch 
	# ## this assumes pytorch is built from source from https://github.com/sparticlesteve/nersc-pytorch-build 
	# module unload python
	# USER="$(basename $HOME)"
	# PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0-gpu
	# source /usr/common/software/python/$PY_VERSION-anaconda-$PY_TIME/etc/profile.d/conda.sh
	# conda activate $PREFIX_PATH
	# export MKLROOT=$PREFIX_PATH
	# BLAS_INC="-I${MKLROOT}/include"	
	# export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
	# BLAS_LIB="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;-lgomp"
	# LAPACK_LIB="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;-lgomp"
	



	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"

	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=fopenmp
	SLU_CUDA_FLAG="-DGPU_ACC -I${CUDA_ROOT}/include"
	STRUMPACK_USE_CUDA=ON
	STRUMPACK_CUDA_FLAGS="-I/global/common/software/m3169/openmpi/4.0.1/gnu-ucx-1.9.0-cuda-10.2.89/include"
	CUBLAS_LIB="${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcudart.so"
	CUBLAS_INCLUDE="${CUDA_ROOT}/include"
# fi 



elif [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
	module unload cray-mpich
	module unload openmpi
	module swap PrgEnv-gnu PrgEnv-intel 
	module swap intel intel/19.0.3.199 
	module load openmpi/4.1.2
	GPTUNEROOT=$PWD
	export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
	BLAS_INC="-I${MKLROOT}/include"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
	BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-liomp5"
	LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-liomp5"
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=qopenmp
# fi 


elif [ $ModuleEnv = 'cori-knl-openmpi-gnu' ]; then
	module load python/$PY_VERSION-anaconda-$PY_TIME
	module unload darshan
	module unload openmpi	
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-intel PrgEnv-gnu
	module load openmpi/4.1.2
	GPTUNEROOT=$PWD
	export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
	BLAS_INC="-I${MKLROOT}/include"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
	BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
	LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=fopenmp

elif [ $ModuleEnv = 'cori-knl-craympich-gnu' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module swap PrgEnv-intel PrgEnv-gnu
	export CRAYPE_LINK_TYPE=dynamic
	GPTUNEROOT=$PWD
	BLAS_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	LAPACK_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	SCALAPACK_LIB="/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so"
	MPICC=cc
	MPICXX=CC
	MPIF90=ftn
	OPENMPFLAG=fopenmp
# fi 

elif [ $ModuleEnv = 'cori-knl-openmpi-intel' ]; then
	module unload darshan
	module unload openmpi
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-gnu PrgEnv-intel 
	# module swap intel intel/19.0.3.199 
	module load openmpi/4.1.2
	GPTUNEROOT=$PWD
	export MKLROOT=/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl
	BLAS_INC="-I${MKLROOT}/include"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_$MKL_TIME/linux/mkl/lib/intel64
	BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-liomp5"
	LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-liomp5"
	SCALAPACK_LIB="$GPTUNEROOT/scalapack-2.1.0/build/lib/libscalapack.so"
	MPICC=mpicc
	MPICXX=mpicxx
	MPIF90=mpif90
	OPENMPFLAG=qopenmp
else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi

export PYTHONPATH=~/.local/cori/$PY_VERSION-anaconda-$PY_TIME/lib/python$PY_VERSION/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

export SCOTCH_DIR=$GPTUNEROOT/examples/STRUMPACK/scotch_6.1.0/install
export ParMETIS_DIR=$GPTUNEROOT/examples/SuperLU_DIST/superlu_dist/parmetis-github
export METIS_DIR=$ParMETIS_DIR
export ButterflyPACK_DIR=$GPTUNEROOT/examples/ButterflyPACK/ButterflyPACK/build/lib/cmake/ButterflyPACK
export STRUMPACK_DIR=$GPTUNEROOT/examples/STRUMPACK/STRUMPACK/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export METIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so;$ParMETIS_DIR/lib/libmetis.so;$ParMETIS_DIR/lib/libGKlib.so"
export METIS_LIBRARIES="$ParMETIS_DIR/lib/libmetis.so;$ParMETIS_DIR/lib/libGKlib.so"



if [[ $ModuleEnv == *"intel"* ]]; then
	rm -rf GPy
	git clone https://github.com/SheffieldML/GPy.git
	cd GPy
	cp ../patches/GPy/coregionalize.py ./GPy/kern/src/.
	cp ../patches/GPy/stationary.py ./GPy/kern/src/.
	cp ../patches/GPy/choleskies.py ./GPy/util/.
	LDSHARED="$MPICC -shared" CC=$MPICC python setup.py build_ext --inplace
	python setup.py install --prefix=$PREFIX_PATH
	cd $GPTUNEROOT
	env CC=$MPICC pip install --user -r requirements_intel.txt
else 
	# scipy>=1.7.0 in requriement.txt doesn't work if --prefix=$PREFIX_PATH is used. Use --user instead. 
	env CC=$MPICC pip install --user -r requirements.txt
fi
# cp ./patches/opentuner/manipulator.py  $PREFIX_PATH/lib/python$PY_VERSION/site-packages/opentuner/search/.

# if openmpi, scalapack needs to be built from source
if [[ $ModuleEnv == *"openmpi"* ]]; then
	wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
	tar -xf scalapack-2.1.0.tgz
	cd scalapack-2.1.0
	rm -rf build
	mkdir -p build
	cd build
	cmake .. \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_INSTALL_PREFIX=. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG " \
		-DBLAS_LIBRARIES="${BLAS_LIB}" \
		-DLAPACK_LIBRARIES="${LAPACK_LIB}"
	make -j32
fi




cd $GPTUNEROOT
rm -rf build
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-$OPENMPFLAG" \
	-DCMAKE_C_FLAGS="-$OPENMPFLAG" \
	-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG " \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$MPICXX \
	-DCMAKE_C_COMPILER=$MPICC \
	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
	-DGPTUNE_INSTALL_PATH="${PREFIX_PATH}/lib/python$PY_VERSION/site-packages"
make install
# cp lib_gptuneclcm.so ../.
# cp pdqrdriver ../

if [[ $BuildExample == 1 ]]; then

	cd $GPTUNEROOT/examples/SuperLU_DIST
	rm -rf superlu_dist
	git clone https://github.com/xiaoyeli/superlu_dist.git
	cd superlu_dist

	#### the following server is often down, so switch to the github repository 
	# wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
	# tar -xf parmetis-4.0.3.tar.gz
	# cd parmetis-4.0.3/
	# cp $GPTUNEROOT/patches/parmetis/CMakeLists.txt .
	# mkdir -p install
	# make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	# make install > make_parmetis_install.log 2>&1
	# cd ../
	# cp $PWD/parmetis-4.0.3/build/Linux-ppc64le/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	# cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


	mkdir -p $ParMETIS_DIR
	rm -f GKlib
	git clone https://github.com/KarypisLab/GKlib.git
	cd GKlib
	make config prefix=$ParMETIS_DIR
	make -j8
	make install
	sed -i "s/-DCMAKE_VERBOSE_MAKEFILE=1/-DCMAKE_VERBOSE_MAKEFILE=1 -DBUILD_SHARED_LIBS=ON/" Makefile
	make config prefix=$ParMETIS_DIR
	make -j8
	make install

	cd ../
	rm -rf METIS
	git clone https://github.com/KarypisLab/METIS.git
	cd METIS
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	make -j8
	make install
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR 
	make -j8
	make install	
	cd ../
	rm -rf ParMETIS
	git clone https://github.com/KarypisLab/ParMETIS.git
	cd ParMETIS
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR shared=1
	make -j8
	make install
	make config cc=$MPICC prefix=$ParMETIS_DIR gklib_path=$ParMETIS_DIR
	make -j8
	make install
	cd ..
	
	mkdir -p build
	cd build
	rm -rf CMakeCache.txt
	rm -rf DartConfiguration.tcl
	rm -rf CTestTestfile.cmake
	rm -rf cmake_install.cmake
	rm -rf CMakeFiles
	cmake .. \
		-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
		-DCMAKE_C_FLAGS="-DXSDK_INDEX_SIZE=64 -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 ${SLU_CUDA_FLAG}" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
	 	-DCMAKE_INSTALL_PREFIX=. \
	 	-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB};${CUBLAS_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
	make pddrive_spawn
	make pzdrive_spawn
	make pddrive3d
	make install


	# cd $GPTUNEROOT/examples/Hypre
	# rm -rf hypre
	# git clone https://github.com/hypre-space/hypre.git
	# cd hypre/src/
	# git checkout v2.19.0
	# ./configure CC=$MPICC CXX=$MPICXX FC=$MPIF90 CFLAGS="-DTIMERUSEMPI" --enable-shared
	# make
	# cp ../../hypre-driver/src/ij.c ./test/.
	# make test


	cd $GPTUNEROOT/examples/ButterflyPACK
	rm -rf ButterflyPACK
	git clone https://github.com/liuyangzhuan/ButterflyPACK.git
	cd ButterflyPACK
	git clone https://github.com/opencollab/arpack-ng.git
	cd arpack-ng
	git checkout f670e731b7077c78771eb25b48f6bf9ca47a490e
	mkdir -p build
	cd build
	cmake .. \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_INSTALL_PREFIX=. \
		-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DCMAKE_Fortran_FLAGS="-$OPENMPFLAG" \
		-DBLAS_LIBRARIES="${BLAS_LIB}" \
		-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DMPI=ON \
		-DEXAMPLES=ON \
		-DCOVERALLS=OFF 
	make
	cd ../../
	mkdir build
	cd build
	cmake .. \
		-DCMAKE_Fortran_FLAGS="-DMPIMODULE $BLAS_INC"\
		-DCMAKE_CXX_FLAGS="" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_INSTALL_PREFIX=. \
		-DCMAKE_INSTALL_LIBDIR=./lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
		-DTPL_ARPACK_LIBRARIES="$PWD/../arpack-ng/build/lib/libarpack.so;$PWD/../arpack-ng/build/lib/libparpack.so"
	make -j32
	make install -j32



	# cd $GPTUNEROOT/examples/STRUMPACK
	# rm -rf scotch_6.1.0
	# wget --no-check-certificate https://gforge.inria.fr/frs/download.php/file/38352/scotch_6.1.0.tar.gz
	# tar -xf scotch_6.1.0.tar.gz
	# cd ./scotch_6.1.0
	# mkdir install
	# cd ./src
	# cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
	# sed -i "s/-DSCOTCH_PTHREAD//" Makefile.inc
	# sed -i "s/-DIDXSIZE64/-DIDXSIZE32/" Makefile.inc
	# sed -i "s/CCD/#CCD/" Makefile.inc
	# printf "CCD = $MPICC\n" >> Makefile.inc
	# sed -i "s/CCP/#CCP/" Makefile.inc
	# printf "CCP = $MPICC\n" >> Makefile.inc
	# sed -i "s/CCS/#CCS/" Makefile.inc
	# printf "CCS = $MPICC\n" >> Makefile.inc
	# cat Makefile.inc
	# make ptscotch 
	# make prefix=../install install


	# cd ../../
	# rm -rf STRUMPACK
	# git clone https://github.com/pghysels/STRUMPACK.git
	# cd STRUMPACK
	# #git checkout 959ff1115438e7fcd96b029310ed1a23375a5bf6  # head commit has compiler error, requiring fixes
	# git checkout 09fb3626cb9d7482528fce522dedad3ad9a4bc9d
	# cp ../STRUMPACK-driver/src/testPoisson3dMPIDist.cpp examples/sparse/. 
	# cp ../STRUMPACK-driver/src/KernelRegressionMPI.py examples/dense/. 
	# chmod +x examples/dense/KernelRegressionMPI.py
	# mkdir build
	# cd build



	# cmake ../ \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_INSTALL_PREFIX=../install \
	# 	-DCMAKE_INSTALL_LIBDIR=../install/lib \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_C_COMPILER=$MPICC \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DSTRUMPACK_COUNT_FLOPS=ON \
	# 	-DSTRUMPACK_TASK_TIMERS=ON \
	# 	-DSTRUMPACK_USE_CUDA=${STRUMPACK_USE_CUDA} \
	# 	-DTPL_CUBLAS_LIBRARIES="${CUBLAS_LIB}" \
	# 	-DTPL_CUBLAS_INCLUDE_DIRS="${CUBLAS_INCLUDE}" \
	# 	-DCMAKE_CUDA_FLAGS="${STRUMPACK_CUDA_FLAGS}" \
	# 	-DTPL_ENABLE_SCOTCH=ON \
	# 	-DTPL_ENABLE_ZFP=OFF \
	# 	-DTPL_ENABLE_PTSCOTCH=ON \
	# 	-DTPL_ENABLE_PARMETIS=ON \
	# 	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	# 	-DTPL_BLAS_LIBRARIES="${BLAS_LIB};$ParMETIS_DIR/lib/libGKlib.so" \
	# 	-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

	# make install -j32
	# make examples -j32


	# cd $GPTUNEROOT/examples/MFEM
	# cp -r $GPTUNEROOT/examples/Hypre/hypre .    # mfem requires hypre location to be here
	# git clone https://github.com/mfem/mfem.git
	# cd mfem
	# cp ../mfem-driver/src/CMakeLists.txt ./examples/.
	# cp ../mfem-driver/src/ex3p_indef.cpp ./examples/.
	# rm -rf mfem-build
	# mkdir mfem-build
	# cd mfem-build
	# cmake .. \
	# 	-DCMAKE_BUILD_TYPE=Release \
	# 	-DCMAKE_CXX_COMPILER=$MPICXX \
	# 	-DCMAKE_CXX_FLAGS="-std=c++11" \
	# 	-DCMAKE_Fortran_COMPILER=$MPIF90 \
	# 	-DBUILD_SHARED_LIBS=ON \
	# 	-DMFEM_USE_MPI=YES \
	# 	-DCMAKE_INSTALL_PREFIX=../install \
	# 	-DCMAKE_INSTALL_LIBDIR=../install/lib \
	# 	-DMFEM_USE_METIS_5=YES \
	# 	-DMFEM_USE_OPENMP=YES \
	# 	-DMFEM_THREAD_SAFE=ON \
	# 	-DMFEM_USE_STRUMPACK=YES \
	# 	-DBLAS_LIBRARIES="${BLAS_LIB}" \
	# 	-DLAPACK_LIBRARIES="${LAPACK_LIB}" \
	# 	-DMETIS_DIR=${METIS_INCLUDE_DIRS} \
	# 	-DMETIS_LIBRARIES="${METIS_LIBRARIES}" \
	# 	-DSTRUMPACK_INCLUDE_DIRS="${STRUMPACK_DIR}/include;${METIS_INCLUDE_DIRS};${PARMETIS_INCLUDE_DIRS};${SCOTCH_DIR}/include" \
	# 	-DSTRUMPACK_LIBRARIES="${STRUMPACK_DIR}/lib/libstrumpack.so;${ButterflyPACK_DIR}/../../../lib/libdbutterflypack.so;${ButterflyPACK_DIR}/../../../lib/libzbutterflypack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libparpack.so;${ButterflyPACK_DIR}/../../../../arpack-ng/build/lib/libarpack.so;${PARMETIS_LIBRARIES};${SCALAPACK_LIB}"
	# make -j32 VERBOSE=1
	# make install
	# make ex3p_indef


	# cd $GPTUNEROOT/examples/IMPACT-Z
	# rm -rf IMPACT-Z
	# git clone https://github.com/impact-lbl/IMPACT-Z.git
	# cd IMPACT-Z
	# git checkout f98eedd2afe8b7e9f20bb72831496b66def334b7  # the Jun 2021 commit that GPTune was able to run
	# cp ../impact-z-driver/*.f90 ./src/Contrl/.
	# mkdir -p build 
	# cd build
	# cmake ../src -DUSE_MPI=ON -DCMAKE_Fortran_COMPILER=$MPIF90 -DCMAKE_BUILD_TYPE=Release 
	# make
	# # mpirun -n 4 ./ImpactZexe-mpi 0 0 0 0 0

fi


cd $GPTUNEROOT
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install --prefix=$PREFIX_PATH
# env CC=mpicc pip install --user -e .								  



cd $GPTUNEROOT
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
cp ../patches/scikit-optimize/space.py skopt/space/.
python setup.py build 
python setup.py install --prefix=$PREFIX_PATH
# env CC=mpicc pip install --user -e .								  


cd $GPTUNEROOT
rm -rf cGP
git clone https://github.com/gptune/cGP
cd cGP/
python setup.py install --prefix=$PREFIX_PATH


cd $GPTUNEROOT
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
# cp ../patches/autotune/problem.py autotune/.
env CC=$MPICC pip install --prefix=$PREFIX_PATH -e .

cd $GPTUNEROOT
rm -rf hybridMinimization
git clone https://github.com/gptune/hybridMinimization.git
cd hybridMinimization/
python setup.py install --prefix=$PREFIX_PATH


cd $GPTUNEROOT

