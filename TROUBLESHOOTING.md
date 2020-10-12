# Troubleshooting

 - Q: What kind of compilers do I need? 
 - A: Usually the Linux system would come with gcc compilers, however, you might need to install gfortran:
 
 `sudo apt-get install gfortran`
 
 - Q: I am not sure if my machine is installed with OpenMPI. How could I check?
 - A: Use following commands in terminal to check the installed version of OpenMPI and compilers.
```
mpicxx --showme:version
mpicc --showme:version
mpif90 --showme:version
```
 - Q: How do I add OpenMPI to PATH?
 - A: Check this entry https://www.open-mpi.org/faq/?category=running#adding-ompi-to-path
 - Q: How do I determine the environment variables given by `EXPORT`?
 - A: Use following commands in terminal to check the installed path, for example, mpicc compiler. Do NOT use gcc without mpi wrapper, which will lead to [mpi.h not found and flag errors](https://stackoverflow.com/questions/26920083/fatal-error-mpi-h-no-such-file-or-directory-include-mpi-h).
 ```
 which mpicc
 export MPICXX=path-to-cxx-compiler-wrapper
```
 - Q: How to find LAPACK/BLAS path?
 - A: Use following commands in terminal to check the installed path, 
 ```
dpkg -L liblapack3
dpkg -L libblas-dev
```
You need to locate the `liblapack.so` and `libblas.so` files (or [so.3 files](https://serverfault.com/questions/401762/solaris-what-is-the-difference-between-so-and-so-1-files/402595#402595) ). If you cannot find it in terminal, use GUI search function.
 
 - Q: What bash command should I type?
 - A: A typical bash command for setting up installation environment is
 ```
export MPICC=/usr/bin/mpicc
export MPICXX=/usr/bin/mpicxx
export MPIF90=/usr/bin/mpif90
export MPIRUN=/usr/bin/mpirun
export BLAS_LIB=/usr/lib/x86_64-linux-gnu/libblas.so
export LAPACK_LIB=/usr/lib/x86_64-linux-gnu/liblapack.so
export GPTUNEROOT=~/GPTune
 ```

