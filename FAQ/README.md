# Frequently Asked Questions

This page contains a collection of frequently asked questions regarding installation and usage of GPTune. For more details, please see the [tutorial material](https://gptune.lbl.gov/documentation/gptune-tutorial-ecp2021/) and the [user manual](https://github.com/gptune/GPTune/blob/master/Doc/GPTune_UsersGuide.pdf).  

Table of Contents
=================

* [Frequently Asked Questions](#frequently-asked-questions)
* [Table of Contents](#table-of-contents)
   * [General Questions](#general-questions)
      * [What applications can be tuned?](#what-applications-can-be-tuned)
   * [Installation](#installation)
      * [Which build script should I use to install GPTune?](#which-build-script-should-i-use-to-install-gptune)
      * [It's not easy to install all dependencies of GPTune correctly, is there a simpler way?](#its-not-easy-to-install-all-dependencies-of-gptune-correctly-is-there-a-simpler-way)
      * [I cannot install GPTune correctly with build scripts or Spack, is there an alternative?](#i-cannot-install-gptune-correctly-with-build-scripts-or-spack-is-there-an-alternative)
      * [I also don't want to use Docker for my production run, is there an alternative?](#i-also-dont-want-to-use-docker-for-my-production-run-is-there-an-alternative)
   * [Usage-Spawning Mode](#usage-spawning-mode)
      * [Do I need to modify my application code if it's not MPI-based?](#do-i-need-to-modify-my-application-code-if-its-not-mpi-based)
   * [Usage-Reverse Communication Interface(RCI) Mode](#usage-reverse-communication-interfacerci-mode)
      * [What is jq?](#what-is-jq)
   * [Known issues](#known-issues)
      * [Cori: runtime error: "Error in `python': break adjusted to free malloc space: 0x0000010000000000 ***"](#cori-runtime-error-error-in-python-break-adjusted-to-free-malloc-space-0x0000010000000000-)
      * [Cori: runtime error: "_pmi_alps_init:alps_get_placement_info returned with error -1"](#cori-runtime-error-_pmi_alps_initalps_get_placement_info-returned-with-error--1)
      * [Hanging at "MLA iteration:  0"](#hanging-at-mla-iteration--0)
      * [Runtime error: "ImportError: cannot import name '_centered' from 'scipy.signal.signaltools'"](#runtime-error-importerror-cannot-import-name-_centered-from-scipysignalsignaltools)
      * [Installation error: "Could not find a version that satisfies the requirement pygmo (from versions: none)"](#installation-error-could-not-find-a-version-that-satisfies-the-requirement-pygmo-from-versions-none)
      * [Runtime error: "ModuleNotFoundError: No module named 'fn'"](#runtime-error-modulenotfounderror-no-module-named-fn)
      * [Runtime error: "module openturns has no module RandomSample"](#runtime-error-module-openturns-has-no-module-randomsample)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## General Questions

### What applications can be tuned?
GPTune can work with essentially any shared-memory/distributed-memory/GPU-based applications on Linux or Mac OS systems. 

Note for distributed-memory applications: If the application is OpenMPI compiled, one can trivially modify your application driver to use the MPI spawning mode, which supports all advanced features in GPTune. If the application is CrayMPICH/Spectrum-MPI/Intel-MPI compiled or even non-MPI (e.g., UPC++/CHARM++), one can use the Reverse Communication Interface(RCI) Mode, which does not require modification of the application code but may not support all GPTune features.  

## Installation
### Which build script should I use to install GPTune?
There are a few example build scripts config_*.sh available under the root directory of GPTune. Depending on what machines you are running the application on, you may need to modify one of the following scripts:  
```
config_cori.sh  # NERSC Cori Haswell/KNL/GPU nodes  
config_wsl.sh   # Linux sub-systems on Windows  
config_summit.sh # Summit machine at ORNL  
config_macbook.zsh or config_macbook_catalina.zsh # Macbook or IMac   
config_cleanlinux.sh # Local Ubuntu or Debian systems  
```
### It's not easy to install all dependencies of GPTune correctly, is there a simpler way?
Instead of installing the software dependencies using the provided sample scripts config_*.sh, one can also consider using Spack (https://spack.io/). GPTune is now available in the develop branch of the spack github repo. To install and test GPTune using Spack. One simply needs e.g., one of the following:  
```
spack install gptune@master~mpispawn  # only install the RCI interface  
spack install gptune@master  # install both the RCI and spawning interfaces  
spack install gptune@master+hypre  # install the hypre example application together with gptune  
spack install gptune@master+superlu  # install the superlu-dist example application together with gptune  
```
Once installed, one can test the installation with:  
```
spack load gptune  
spack test run gptune  
```
For using spack-installed gptune for your own application:
copy the run_env.sh generated by spack test (typically at ./opt/spack/XXX/XXX/gptune-2.1.0-XXX/.spack/test) to your own directory (that contains the gptune launcher), then
. run_env.sh
and use $MPIRUN to launch GPTune. See https://github.com/gptune/GPTune/blob/master/examples/GPTune-Demo/run_examples.sh for an example.

Note for DOE leadership machines: the spack installation is partially tested on Cori, which requires modified spack package files instead of using those from the spack github repo. Please see https://github.com/liuyangzhuan/spack_gptune_nersc for more details.
### I cannot install GPTune correctly with build scripts or Spack, is there an alternative?
For local machines and small clusters, one can also consider using our pre-built Docker image.  
```
docker pull liuyangzhuan/gptune:4.5 # Image# 4.5 can get out-dated, try look for higher versions.   
docker run -it liuyangzhuan/gptune:4.5
```
Once the docker image is launched, one can test the image with:
```
cd /app/GPTune/  
edit run_examples.sh to select which applications to test  
bash run_examples.sh
```
### I also don't want to use Docker for my production run, is there an alternative?
Yes, there is now a light-weight version of GPTune, which doesn't rely on mpi4py, openturns, pygmo and openmpi 4.0+. But you do need a working MPI (e.g. mpich or openmpi<4.0). You need set the following environment variable during the installation and use stages of GPTune:
```
export GPTUNE_LITE_MODE=1
```
See config_cleanlinux.sh or config_macbook.zsh as an example. Note that when the GPTune_lite mode is used, certain GPTune features cannot be used. 
## Usage-Spawning Mode
The spawning mode of GPTune relies on the mpi spawning mechanism (only available in OpenMPI) to launch a parallel application from Python codes. As a mandatory requirement, both GPTune and the application need to be compiled with OpenMPI.  
### Do I need to modify my application code if it's not MPI-based?
For non-distributed-memory application code, one doesn't need to modify the driver code. That said, one always need to either define a Python function that reads the tuning parameters, launch the application code, and get the code output (the Spawning mode), or use a bash script to query GPTune for next sample points, launch the application, grep the results from runlogs/data files and use jq to write the results into the GPTune database file (the RCI mode).  
## Usage-Reverse Communication Interface(RCI) Mode
For non-OpenMPI-based applications, one can use the RCI mode of GPTune, which essentially relies on a bash script to query GPTune for next samples, launch the applications as one normally does from command line (e.g, using srun/jsrun/mpirun), and write back results to the GPTune database file, and query GPTune again.   
### What is jq?
jq is a Linux command line utility that is easily used to extract/write data from/into JSON documents. The RCI mode relies on jq to update the GPTune database. 

## Known issues
### Cori: runtime error: "Error in `python': break adjusted to free malloc space: 0x0000010000000000 ***"
One needs to unload the craype-hugepages2M module with: module unload craype-hugepages2M
### Cori: runtime error: "_pmi_alps_init:alps_get_placement_info returned with error -1"
Intead of "python xx.py", one needs "srun/mpirun -n 1 xx.py" to get the correct MPI infrastructure.
### Hanging at "MLA iteration:  0"
This typically means that a wrong version of openmpi is used at runtime. Make sure to use the same openmpi version as the one for gptune installation.
### Runtime error: "ImportError: cannot import name '_centered' from 'scipy.signal.signaltools'"
This is due to the use of scipy-1.8.0 (or newer) and statsmodels-0.12.2 (or older). You can upgrade statsmodels to 0.13.2. 
### Installation error: "Could not find a version that satisfies the requirement pygmo (from versions: none)"
For python3.9+ pip install pygmo doesn't work. You need to install pygmo from source. Assume that your system has BOOST(>=1.69) installed at 'BOOST_ROOT' and pybind11 installed at 'SITE_PACKAGE'/pybind11. The C and C++ compilers are 'MPICC' and 'MPICXX'. If you don't have BOOST>=1.69, you can install it from source:
```
cd $GPTUNEROOT  
rm -rf download
wget -c 'http://sourceforge.net/projects/boost/files/boost/1.69.0/boost_1_69_0.tar.bz2/download'  
tar -xvf download  
cd boost_1_69_0/  
./bootstrap.sh --prefix=$PWD/build  
./b2 install  
export BOOST_ROOT=$GPTUNEROOT/boost_1_69_0/build 

```
The following lines install TBB, pagmo and pygmo from source:
```
export TBB_ROOT=$GPTUNEROOT/oneTBB/build  
export pybind11_DIR=$SITE_PACKAGE/pybind11/share/cmake/pybind11  
export BOOST_ROOT=XXX  # if insalled from source, remove this line
export pagmo_DIR=$GPTUNEROOT/pagmo2/build/lib/cmake/pagmo  

cd $GPTUNEROOT  
rm -rf oneTBB  
git clone https://github.com/oneapi-src/oneTBB.git  
cd oneTBB  
mkdir build  
cd build  
cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_INSTALL_LIBDIR=$PWD/lib -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  
make -j16  
make install  
git clone https://github.com/wjakob/tbb.git  
cp tbb/include/tbb/tbb_stddef.h include/tbb/.  

cd $GPTUNEROOT  
rm -rf pagmo2  
git clone https://github.com/esa/pagmo2.git  
cd pagmo2
git checkout 1d41b1b5f70e59db8481ff8e6213f06f3b8b51f2  
mkdir build  
cd build  
cmake ../ -DCMAKE_INSTALL_PREFIX=$PWD -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX -DCMAKE_INSTALL_LIBDIR=$PWD/lib  
make -j16  
make install  
cp lib/cmake/pagmo/*.cmake .  

cd $GPTUNEROOT  
rm -rf pygmo2  
git clone https://github.com/esa/pygmo2.git  
cd pygmo2  
mkdir build  
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX_PATH -DPYGMO_INSTALL_PATH="$SITE_PACKAGE" -DCMAKE_C_COMPILER=$MPICC -DCMAKE_CXX_COMPILER=$MPICXX  
make -j16  
make install  
```
Note that, if you have installed TBB, pagmo, and pygmo2 from source codes with the installation paths described in the above instructions, then, to use the pygmo module in GPTune, you may need to provide the path to the TBB and pagmo library, for example:
```
export LD_LIBRARY_PATH=$GPTUNEROOT/pagmo2/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GPTUNEROOT/oneTBB/build/lib:$LD_LIBRARY_PATH
```

### Runtime error: "ModuleNotFoundError: No module named 'fn'"
This is due to the dependency on fn from opentuner. Note fn has been removed from opentuner (https://github.com/jansel/opentuner/pull/155). Make sure you use the patch file provided by GPTune, as
```
cp $GPTUNEROOT/patches/opentuner/manipulator.py your-python-site-packages/opentuner/search/.
```
### Runtime error: "module openturns has no module RandomSample"
This is typically due to the fact that openturns installation is incomplete as it requires TBB, which only works on AMD and Intel-based architectures. You can swith openturns off by setting options['sample_class'] = 'SampleLHSMDU'. 

