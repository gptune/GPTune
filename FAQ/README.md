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
   * [Usage-Spawning Mode](#usage-spawning-mode)
      * [Do I need to modify my application code if it's not MPI-based?](#do-i-need-to-modify-my-application-code-if-its-not-mpi-based)
   * [Usage-Reverse Communication Interface(RCI) Mode](#usage-reverse-communication-interfacerci-mode)
      * [What is jq?](#what-is-jq)
   * [Known issues](#known-issues)
      * [Cori: runtime error: "Error in `python': break adjusted to free malloc space: 0x0000010000000000 ***"](#cori-runtime-error-error-in-python-break-adjusted-to-free-malloc-space-0x0000010000000000-)
      * [Cori: runtime error: "_pmi_alps_init:alps_get_placement_info returned with error -1"](#cori-runtime-error-_pmi_alps_initalps_get_placement_info-returned-with-error--1)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

## General Questions

### What applications can be tuned?
GPTune can work with essentially any shared-memory/distributed-memory/GPU-based applications on Linux or Mac OS systems. 

Note for distributed-memory applications: If the application is OpenMPI compiled, one can trivially modify your application driver to use the MPI spawning mode, which supports all advanced features in GPTune. If the application is CrayMPICH/Spectrum-MPI/Intel-MPI compiled or even non-MPI (e.g., UPC++/CHARM++), one can use the Reverse Communication Interface(RCI) Mode, which does not require modification of the application code but may not support all GPTune features.  

## Installation
### Which build script should I use to install GPTune?
There are a few example build scripts config_*.sh available under the root directory of GPTune. Depending on what machines you are running the application on, you may need to modify one of the following scripts:  

config_cori.sh  # NERSC Cori Haswell/KNL/GPU nodes  
config_wsl.sh   # Linux sub-systems on Windows  
config_summit.sh # Summit machine at ORNL  
config_macbook.zsh or config_macbook_catalina.zsh # Macbook or IMac   
config_cleanlinux.sh # Local Ubuntu or Debian systems  

### It's not easy to install all dependencies of GPTune correctly, is there a simpler way?
Instead of installing the software dependencies using the provided sample scripts config_*.sh, one can also consider using Spack (https://spack.io/). GPTune is now available in the develop branch of the spack github repo. To install and test GPTune using Spack. One simply needs e.g., one of the following:  

spack install gptune@master~mpispawn  # only install the RCI interface  
spack install gptune@master  # install both the RCI and spawning interfaces  
spack install gptune@master+hypre  # install the hypre example application together with gptune  
spack install gptune@master+superlu  # install the superlu-dist example application together with gptune  

Once installed, one can test the installation with:  

spack load gptune  
spack test run gptune  

Note for DOE leadership machines: the spack installation is partially tested on Cori, which requires modified spack package files instead of using those from the spack github repo. If you are interested, please contact GPTune developers. 
### I cannot install GPTune correctly with build scripts or Spack, is there an alternative?
For local machines and small clusters, one can also consider using our pre-built Docker image.  

docker pull liuyangzhuan/gptune:2.6 # Image# 2.6 is getting out-dated, try look for higher versions.   
docker run -it liuyangzhuan/gptune:2.6

Once the docker image is launched, one can test the image with:

cd /app/GPTune/  
edit run_examples.sh to select which applications to test  
bash run_examples.sh

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


