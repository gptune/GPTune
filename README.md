<p align="center">
<img src="GPTune_icon.png" width="200">
</p>

# [**GPTune**](https://gptune.lbl.gov) 

*GPTune* is an autotuning framework that relies on multitask and transfer learnings to help solve the underlying black-box optimization problem using Bayesian optimization methodologies.

Table of Contents
=================

* [<a href="https://gptune.lbl.gov" rel="nofollow"><strong>GPTune</strong></a>](#gptune)
* [Table of Contents](#table-of-contents)
   * [Features](#features)
   * [Installation](#installation)
      * [Installation using example scripts](#installation-using-example-scripts)
         * [Ubuntu/Debian-like systems supporting apt-get](#ubuntudebian-like-systems-supporting-apt-get)
         * [Mac OS supporting homebrew](#mac-os-supporting-homebrew)
         * [NERSC Cori](#nersc-cori)
         * [NERSC Perlmutter](#nersc-perlmutter)
         * [OLCF Summit](#olcf-summit)
      * [Installation using spack](#installation-using-spack)
      * [Installation from scratch](#installation-from-scratch)
      * [Install OpenMPI](#install-openmpi)
         * [Install SCALAPACK](#install-scalapack)
         * [Install mpi4py](#install-mpi4py)
         * [Install scikit-optimize](#install-scikit-optimize)
         * [Install autotune](#install-autotune)
         * [Install cGP](#install-cgp)
         * [Install GPTune](#install-gptune)
      * [Using prebuilt docker images](#using-prebuilt-docker-images)
   * [Examples](#examples)
      * [GPTune-Demo](#gptune-demo)
      * [SCALAPACK QR](#scalapack-qr)
      * [SuperLU_DIST](#superlu_dist)
   * [Usage](#usage)
      * [Problem description](#problem-description)
         * [Spaces](#spaces)
         * [Parameters](#parameters)
         * [Constraints](#constraints)
         * [Objective Function](#objective-function)
         * [Performance Models](#performance-models)
      * [GPTune invocation](#gptune-invocation)
   * [Resources](#resources)
   * [References](#references)
      * [Publications](#publications)
      * [BibTeX citation](#bibtex-citation)
      * [Copyright](#copyright)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## Features
- **(High-performance and parallel tuning)** GPTune is designed to tune applications running on large-scale cluster systems and can exploit distributed memory parallelism for accelerating surrogate modeling.
  - [Example: Autotuning ScaLAPACK's PDGEQRF using distributed parallel surrogate modeling](https://github.com/gptune/GPTune/blob/master/examples/Scalapack-PDGEQRF/scalapack_MLA.py)
- **(Multitask learning-based autotuning)** GPTune supports multitask learning-based autotuning that allows us to tune multiple tuning problems simultaneously. Multitask learning would predict an optimal tuning parameter configuration using a fewer number of evaluations compared to single task autotuning by modeling the linear dependence of the multiple tasks.
  - [Example: Multitask learning-based autotuning of ScaLAPACK's PDGEQRF](https://github.com/gptune/GPTune/blob/master/examples/Scalapack-PDGEQRF/scalapack_MLA.py)
- **(Transfer learning-based autotuning)** GPTune supports transfer learning-based autotuning to tune the given tuning task by leveraging already available performance data collected for different tasks. Different tasks can mean different input problem sizes or the same input problem on different machine and/or software settings.
  - [Example: Transfer learning from a different input problem for autotuning ScaLAPACK's PDGEQRF](https://github.com/gptune/GPTune/blob/master/examples/Scalapack-PDGEQRF/scalapack_TLA_task.py)
  - [Example: Transfer learning from a different machine setting for autotuning ScaLAPACK's PDGEQRF](https://github.com/gptune/GPTune/blob/master/examples/Scalapack-PDGEQRF/scalapack_TLA_machine.py)
- **(GPTuneBand (multi-fidelity autotuning))** Multi-fidelity tuning uses multiple fidelity levels to guide sampling (generating many cheap samples from lower-fidelity levels). GPTuneBand combines multitask learning with a multi-armed bandit strategy to guide sampling of the given tuning problem.
  - [Example: Multi-fidelity tuning of STRUMPACK's Kernel Ridge Regression (need to set the tuner option to "GPTuneBand")](https://github.com/gptune/GPTune/blob/master/examples/STRUMPACK/strumpack_MLA_KRR_MB.py)
- **(Multi-objective tuning)** Beyond the classical single-objective tuning, GPTune supports multi-objective tuning that uses NSGA2 algorithm to maximize multiple EI functions for multiple objectives. For an objective, users can also specify whether they want to optimize (minimize) the objective within the given range, or they just want the objective is within the given range.
  - [Example: Multi-objective tuning of SuperLU_DIST](https://github.com/gptune/GPTune/blob/master/examples/SuperLU_DIST/superlu_MLA_MO.py)
- **(Unified interface for different autotuners)** GPTune uses a unified Python interface and supports using several different autotuners.
  - [Example: Comparing GPTune, HpBandSter, and OpenTuner for a synthetic function](https://github.com/gptune/GPTune/blob/master/examples/GPTune-Demo/demo_comparetuners.py)
- **(History database)** We provide an autotuning database called GPTune history database which allows users to save and re-use performance data to reduce the cost of the expensive black-box objective function. The history database enables several useful autotuning capabilities. The details are outlined in the GPTune and history database project webpage at [https://gptune.lbl.gov/about](https://gptune.lbl.gov/about).

GPTune is part of the xSDK4ECP effort supported by the Exascale Computing Project (ECP).
Our GPTune website at https://gptune.lbl.gov provides a shared database repository where the user can share their tuning performance data with other users.

## Installation

### Installation using nix (for single-node systems)

Nix may be used to install GPTune and all its dependencies on single-node systems, including personal computers and cloud servers (both with and without root access). Nix pulls in independent copies of GPTune's dependencies, and as a result it will neither affect nor be affected by the state of your system's packages.
#### 1. Install Nix

**If you have root access,** run this command to automatically install Nix, then immediately proceed to step 2:

```sh <(curl -L https://nixos.org/nix/install) --daemon``` 

For more details, see the [manual](https://nixos.org/manual/nix/stable/installation/installing-binary.html) for full details).

**If you do *not* have root access,** you can install Nix as an unpriviliged user using one of [these] methods. For systems supporting user namespaces (follow the instructions [here](https://github.com/nix-community/nix-user-chroot#check-if-your-kernel-supports-user-namespaces-for-unprivileged-users) to check for user namespace support), including Debian, Ubuntu, and Arch, [nix-user-chroot](https://github.com/nix-community/nix-user-chroot) is recommended; the following steps may be used to install it. First, download the appropriate [static binary](https://github.com/nix-community/nix-user-chroot/releases) for your hardware platform:

```
#replace the link below with the appropriate build for your architecture
wget -O nix-user-chroot https://github.com/nix-community/nix-user-chroot/releases/download/1.2.2/nix-user-chroot-bin-1.2.2-x86_64-unknown-linux-musl
chmod +x nix-user-chroot
#optionally, add nix-user-chroot to $PATH - you'll be running it a lot
```

Then, select an installation location. For this example, we will use `~/.nix`. Note that Nix will perform a significant amount of disk I/O to this location, so make sure that this directory is not located on a network drive (NFS, etc.) or builds may be slowed by up to an order of magnitude (for example, we recommend that UC Berkeley Millennium cluster users use a folder in `/scratch` or `/nscratch` instead). You may then install nix with:

```
mkdir -m 0755 ~/.nix
./nix-user-chroot ~/.nix bash -c "curl -L https://nixos.org/nix/install | bash"
```

You may now enter the nix chroot environment with

```./nix-user-chroot ~/.nix bash -l```

This works much like a python virtualenv or conda shell - it will drop you into a environment where the Nix package manager and tools you have installed with Nix are available. As with a python virtualenv, you must be inside this environment in order to access tools (e.g. GPTune) that are installed with Nix (i.e. you must run it in each shell where you need these tools). All programs, files, etc. outside the environment should be accessible from within the environment as well.

*Troubleshooting note: if you run into "out of space" errors during builds, set the `TMPDIR` environment variable when you run this command to a location on a disk with plenty of space, e.g. `TMPDIR=/scratch/dinh/tmp nix-user-chroot /scratch/dinh/.nix bash`*

**If you do *not* have root access and your system does *not* support user namespaces**, you can [install Nix using proot](https://nixos.wiki/wiki/Nix_Installation_Guide#PRoot).

#### 2: Enable nix flakes

Nix flakes, which we use to build GPTune, are technically an experimental feature in nix (this is not a mark of instability - flakes have existed and been widely used for years, but remain marked as experimental since there's a slim possibility that their interface might change). As a result, they must be manually enabled, which can be done by adding this line:

```
experimental-features = nix-command flakes
```

to any one (or more) of the following locations (if the file(s) in question don't exist, feel free to create them):

- `~/.config/nix/nix.conf` (recommended, affects your user account only)
- `/etc/nix/nix.conf` (if you have root access, and want to enable flakes for everyone on your system)
- `/nix/etc/nix/nix.conf` (for chroot-based installs only)

#### 3: Build GPTune

Clone the GPTune repo and cd into its directory:

```
git clone https://github.com/gptune/GPTune
cd GPTune
```

then run

```nix develop```

to enter an environment where the `python` executable has all the dependencies needed.

Alternatively, if you just want the C++ libraries for GPTune (e.g. to link with), run `nix build .#gptune-libs`, which will put the librarires in `result/gptune`.
### Installation using example scripts
The following example build scripts are available for a collection of tested systems. 

#### Ubuntu/Debian-like systems supporting apt-get
The following script installs everything from scratch and can take up to 2 hours depending on the users' machine specifications. If "MPIFromSource=0", you need to set PATH, LIBRARY_PATH, LD_LIBRARY_PATH and MPI compiler wrappers when prompted.
```
config_cleanlinux.sh
```

#### Mac OS supporting homebrew
The following script installs everything from scratch and can take up to 2 hours depending on the users' machine specifications. The user may need to set pythonversion, gccversion, openblasversion, lapackversion on the top of the script to the versions supported by your homebrew software. 
```
config_macbook.zsh
```

#### NERSC Cori
The following script installs GPTune with mpi, python, compiler and cmake modules on Cori. Note that you can set "proc=haswell or knl", "mpi=openmpi or craympich" and "compiler=gnu or intel". Setting mpi=craympich will limit certain GPTune features. Particularly, only the so-called reverse communication interface (RCI) mode can be used, please refer to the user guide for details https://github.com/gptune/GPTune/blob/master/Doc/GPTune_UsersGuide.pdf.
```
config_cori.sh
```

#### NERSC Perlmutter
The following script installs GPTune with mpi, python, compiler, cudatoolkit and cmake modules on Perlmutter. Note that you need to set "proc=milan #(CPU nodes) or gpu #(GPU nodes)", "mpi=openmpi or craympich" and "compiler=gnu". Setting mpi=craympich will only support the RCI mode.
```
config_perlmutter.sh
```

#### OLCF Summit
The following script installs GPTune with mpi, python, compiler, cuda and cmake modules on Summit. Note that you can set "proc=power9", "mpi=spectrummpi" and "compiler=gnu". Currently, only the RCI mode can be used on Summit.
```
config_summit.sh
```


### Installation using spack
One can also consider using Spack (https://spack.io/). To install and test GPTune using Spack (the develop branch of the spack github repo is highly recommended), one simply needs:
```
spack install gptune@master
spack load gptune@master
```

### Installation from scratch
GPTune relies on OpenMPI (4.0 or higher) or CrayMPICH (8.1.23 or higher), Python (3.7 or higher), BLAS/LAPACK, SCALAPACK (2.1.0 or higher), mpi4py, scikit-optimize, cGP and autotune, which need to be installed by the user. In what follows, we assume Python, BLAS/LAPACK have been installed (with the same compiler version):
```
export MPICC=path-to-c-compiler-wrapper  # see next subsection to install OpenMPI from source if one doesn't yet have one installed. 
export MPICXX=path-to-cxx-compiler-wrapper
export MPIF90=path-to-f90-compiler-wrapper
export MPIRUN=path-to-mpirun
export BLAS_LIB=path-to-blas-lib
export LAPACK_LIB=path-to-lapack-lib
export GPTUNEROOT=path-to-gptune-root-directory
export SITE_PACKAGES_PATH=path-to-your-site-packages
```

The rest can be installed as follows:
### Install OpenMPI
```
cd $GPTUNEROOT
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
bzip2 -d openmpi-4.0.1.tar.bz2
tar -xvf openmpi-4.0.1.tar 
cd openmpi-4.0.1/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$MPICC CXX=$MPICXX F77=$MPIF90 FC=$MPIF90 --enable-mpi1-compatibility --disable-dlopen
make -j4
make install
export PATH=$PATH:$GPTUNEROOT/openmpi-4.0.1/bin
export MPICC="$GPTUNEROOT/openmpi-4.0.1/bin/mpicc"
export MPICXX="$GPTUNEROOT/openmpi-4.0.1/bin/mpicxx"
export MPIF90="$GPTUNEROOT/openmpi-4.0.1/bin/mpif90"
export LD_LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GPTUNEROOT/openmpi-4.0.1/lib:$LIBRARY_PATH 
```

#### Install SCALAPACK
```
cd $GPTUNEROOT
wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
mkdir -p install
cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_Fortran_FLAGS="-fopenmp" \
    -DBLAS_LIBRARIES="$BLAS_LIB" \
    -DLAPACK_LIBRARIES="$LAPACK_LIB"
make 
make install
export SCALAPACK_LIB="$PWD/install/lib/libscalapack.so"  
```


#### Install mpi4py
```
cd $GPTUNEROOT
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$MPICC -shared"
python setup.py install --user
export PYTHONPATH=$PYTHONPATH:$PWD
```

#### Install scikit-optimize
```
cd $GPTUNEROOT
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
cp ../patches/scikit-optimize/space.py skopt/space/.
pip install --user -e .
export PYTHONPATH=$PYTHONPATH:$PWD
```

#### Install autotune
autotune contains a common autotuning interface used by GPTune and ytopt. It can be installed as follows:
```
cd $GPTUNEROOT
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install --user -e .
export PYTHONPATH=$PYTHONPATH:$PWD
```

#### Install cGP
```
cd $GPTUNEROOT
git clone https://github.com/gptune/cGP
cd cGP/
python setup.py install 
```

#### Install GPTune
GPTune also depends on several external Python libraries as listed in the `requirements.txt` file, including numpy, joblib, scikit-learn, scipy, statsmodels, pyaml, matplotlib, GPy, openturns,lhsmdu, ipyparallel, opentuner, hpbandster, pygmo, filelock, requests, pymoo and cloudpickle. These Python libraries can all be installed through the standard Python repository through the pip tool.
```
cd $GPTUNEROOT
env CC=$MPICC pip install --user -r requirements.txt
```

In addition, the following MPI-enabled component needs to be installed:
```
cd $GPTUNEROOT
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p build
cd build
cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DGPTUNE_INSTALL_PATH=$SITE_PACKAGES_PATH \
    -DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
    -DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
    -DTPL_SCALAPACK_LIBRARIES=$SCALAPACK_LIB
make 
```

### Using prebuilt docker images
One can also try the prebuilt docker image of GPTune to test its functionality 
```
docker pull liuyangzhuan/gptune:4.5
docker run -it -v $HOME:$HOME liuyangzhuan/gptune:4.5
```

## Examples
There are a few examples included in GPTune, each example is located in a seperate directory ./examples/[application_name]. The user needs to edit examples/[application_name]/.gptune/meta.json to define machine information and software dependency, before running the tuning examples. 

Please take a look at the following two scripts to run the complete examples. Note that these scripts first load ./run_env.sh (the user needs to modify this file to define appropriate runtime variables, machine and software information), generate the examples/[application_name]/.gptune/meta.json file, and then invoke the tuning experiment.  
https://github.com/gptune/GPTune/blob/master/examples/GPTune-Demo/run_examples.sh
https://github.com/gptune/GPTune/blob/master/run_ppopp.sh

### GPTune-Demo
The file `demo.py` in the `examples/GPTune-Demo` folder shows how to describe the autotuning problem for a sequential objective function and how to invoke GPTune 
```
cd $GPTUNEROOT/examples/GPTune-Demo

edit .gptune/meta.json
$MPIRUN -n 1 python ./demo.py

or 
edit ../../run_env.sh
bash run_examples.sh
```
### SCALAPACK QR
The files `scalapack_*.py` in the `examples/Scalapack-PDGEQRF` folder shows how to tune the parallel QR factorization subroutine PDGEQRF with different features of GPTune. 
```
cd $GPTUNEROOT/examples/Scalapack-PDGEQRF

edit .gptune/meta.json
$MPIRUN -n 1  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nprocmin_pernode 1 -ntask 2 -nrun 20 -optimization 'GPTune'

or

edit ../../run_env.sh
bash run_examples.sh
```
### SuperLU_DIST
First, SuperLU_DIST needs to be installed with the same OpenMPI/CrayMPICH and BLAS/LAPACK as the above.
```
cd $GPTUNEROOT/examples
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
make install 
cd ../
export PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
export PARMETIS_LIBRARIES=$PWD/parmetis-4.0.3/install/lib/libparmetis.so
mkdir -p build
cd build
cmake .. \
    -DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
    -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_COMPILER=$MPICXX \
    -DCMAKE_C_COMPILER=$MPICC \
    -DCMAKE_Fortran_COMPILER=$MPIF90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DTPL_BLAS_LIBRARIES="$BLAS_LIB" \
    -DTPL_LAPACK_LIBRARIES="$LAPACK_LIB" \
    -DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
    -DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn 
```
Note that `pddrive_spawn` is a modified application driver that will be launched by GPTune via MPI spawning (see the Usage section). The files `superlu_*.py` in the `examples/SuperLU_DIST` folder shows how to tune the performance of sparse LU factorization with different features of GPTune. 
```
cd $GPTUNEROOT/examples/SuperLU_DIST

edit .gptune/meta.json
$MPIRUN -n 1 python ./superlu_MLA_MO.py -nprocmin_pernode 1 -ntask 1 -nrun 10 -optimization 'GPTune'

or

edit ../../run_env.sh
bash run_examples.sh
```


## Usage

### Problem description

#### Spaces

In order to autotune a certain application, three spaces have to be defined through an instance of the **Space** class.
1. Input Space (IS): this space defines the problems that the application targets.
Every point in this space represents one instance of a problem.
In the context of GPTune, the word *task* means application *problem*.
2. Parameter Space (PS): this space defines the application parameters to be tuned.
A point in this space represents a combination of the parameters.
The goal of the tuner is to find the best possible combination that minimizes the objective function of the application.
3. Output Space (OS): this space defines the result(s) of the application, i.e., the objective of the application to be optimized.
For examples, this can be runtime, memory or energy consumption in HPC applications or prediction accuracy in machine learning applications.

#### Parameters

Every dimension of the above mentioned spaces is defined by a **Parameter** object.
Every parameter is defined by its name, type and range or set of values.
Three types of parameters can be defined:
1. Real: defines floating point parameters.
The range of values that the parameter spans should be defined in the *range* argument.
2. Integer: defines integer parameters.
The range of values that the parameter spans should be defined in the *range* argument.
3. Categorical: defines parameters that take their values in a set or list of (string) values.
The list of valid values defining the parameter should be defined in the *values* argument.
(*Note*: If the problems the application targets cannot be defined in a Cartesian space, the user can simply give a list of problems (as a Categorical parameter) in the definition of the task space.)

#### Constraints

Not all points in the task or input spaces correspond to valid problems or parameter configurations.
Constraints might exist that define the validity of a given combination of input parameters and problem description parameters results.
Two ways exist to define constraints in GPTune:
1. Strings: the user can define a Python statement in a string.
The evaluation of that statement should be a boolean.
2. Functions: the user can define a Python function that returns a boolean. 

#### Objective Function
The user need to define a Python function representing the objective function to be optimized
```
def objectives(point)
   # extract the parameters from 'point', invoke the application code, and store the objective function value in 'result'
   return result
```
Here point is a dictionary containing key pairs representing task and tuning parameters. If the application code is distributed-memory parallel, the user needs to modify the application code (in C, C++, Fortran, Python, etc.) with the MPI spawning syntax and invoke the code using MPI.COMM_SELF.Spawn from mpi4py. Please refer to the user guide for details https://github.com/gptune/GPTune/blob/master/Doc/GPTune_UsersGuide.pdf.   

#### Performance Models

The user having additional knowledge about the application can help speed up or improve the result of the tuning process by passing a performance model(s) of the objective function to be optimized.

These models are defined through Python functions following similarly to the constraints definition.

### GPTune invocation

Once the parameters and spaces (and optionally constraints and models) are defined, an object of the **GPTune** class has to be instantiated.
Then, the different kinds of tuning techniques (*MLA, ...*) can be called through it.

## Resources

[**Manual**] [GPTune UsersGuide](https://gptune.lbl.gov/documentation/gptune-user-guide)

[**Manual**] [GPTune History Database & Shared Repository](https://gptune.lbl.gov/docs/index.html)

[**Tutorial**] [GPTune: Performance Autotuner for ECP Applications, Tutorial at ECP Annual Meeting, April 14, 2021](https://gptune.lbl.gov/gptune-tutorial-ecp2021)

[**Talk**] [GPTune: Multitask Learning for Autotuning Exascale Applications, PPoPP 2021, February 27, 2021](https://www.youtube.com/watch?v=QDcZTEKh_b0)

[**Talk**] [Autotuning exascale applications with Gaussain Process Regression, E-NLA Seminar, October 14, 2020](https://www.youtube.com/watch?v=Xnj8FDquMgI&t=287s)

## References
### Publications
Y. Liu, W.M. Sid-Lakhdar, O. Marques, X. Zhu, C. Meng, J.W. Demmel, and X.S. Li. "GPTune: multitask learning for autotuning exascale applications", in Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '21). Association for Computing Machinery, New York, NY, USA, 234–246. DOI:https://doi.org/10.1145/3437801.3441621

### BibTeX citation
```
@inproceedings{liu2021gptune,
  title={GPTune: multitask learning for autotuning exascale applications},
  author={Liu, Yang and Sid-Lakhdar, Wissam M and Marques, Osni and Zhu, Xinran and Meng, Chang and Demmel, James W and Li, Xiaoye S},
  booktitle={Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
  pages={234--246},
  year={2021}
}
```

### Copyright
GPTune Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S.Dept. of Energy) and the University of California, Berkeley.
All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.
