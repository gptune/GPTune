# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department 
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#
# .PHONY: all lib demo clean
.PHONY: all lib clean

compiler_version = gcc
#compiler_version = intel

#mpi_version = sgimpt
mpi_version = openmpi
#mpi_version = intelmpi

CFLAGS= -O3 -Wall -fPIC -std=c11
#CFLAGS= -g -O0 -Wall -fpic
LDFLAGSLIB= -shared
LDFLAGSEXE=
INCS = -I.
MACHINE = $(shell hostname -s)


ifeq ($(compiler_version),gcc)
ifeq ($(MACHINE),cori)
	LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lmkl_blacs_$(mpi_version)_lp64 -lmkl_scalapack_lp64 -lmkl_avx -lmkl_def -lpthread -lm
else 
	LIBS = -L/usr/lib/x86_64-linux-gnu/ -lscalapack -llapack -lblas  	
endif	
	CFLAGS+= -fopenmp
	LDFLAGSEXE+=
	LIBS+= -lgomp
endif
ifeq ($(compiler_version),intel)
	LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lmkl_blacs_$(mpi_version)_lp64 -lmkl_avx -lmkl_def -lpthread -lm
	CFLAGS+= -qopenmp
	LDFLAGSEXE+= -dynamic # -nofor_main
	LIBS+= -liomp5
endif

ifeq ($(mpi_version),sgimpt)
	CC=cc
endif
ifeq ($(mpi_version),openmpi)
	CC=cc#$(OMPI_DIR)/bin/mpicc
endif
ifeq ($(mpi_version),intelmpi)
	CC=cc#$(I_MPI_ROOT)/intel64/bin/mpicc
endif

# all: clean lib demo
all: clean lib

lib: lcm.c
	$(CC) $(CFLAGS) $(INCS) -c lcm.c -o lcm.o;
	$(CC) -o liblcm.so lcm.o $(LDFLAGSLIB) $(LIBS);

# demo: demo.c
	# $(CC) $(CFLAGS) $(INCS) -c demo.c -o demo.o;
	# $(CC) -o demo lcm.o demo.o $(LDFLAGSEXE) $(LIBS);

clean:
	rm -f lcm.o liblcm.so 
	# rm -f lcm.o liblcm.so demo.o demo

