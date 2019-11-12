all: driver

ifndef MACHINE_NAME
MACHINE_NAME=$(NERSC_HOST)
endif

BIN=bin/$(MACHINE_NAME)
EXP=exp/$(MACHINE_NAME)
SRC=src

CC=mpicc
FC=mpif90
CFLAGS= -O3 -Wall -fPIC -std=c11 -fopenmp
LDFLAGS= -shared
LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lmkl_blacs_openmpi_lp64 -lmkl_scalapack_lp64 -lmkl_avx -lmkl_def -lpthread -lm -lgomp

build_dirs:
	mkdir -p $(BIN)
	mkdir -p $(EXP)

# Compile
$(BIN)/pdqrinfo.o:$(SRC)/pdqrinfo.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdqrinfo.o -c $(SRC)/pdqrinfo.f
$(BIN)/pdmatgen.o:$(SRC)/pdmatgen.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdmatgen.o -c $(SRC)/pdmatgen.f
$(BIN)/pdgeqrrv.o:$(SRC)/pdgeqrrv.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdgeqrrv.o -c $(SRC)/pdgeqrrv.f
$(BIN)/pdlafchk.o:$(SRC)/pdlafchk.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdlafchk.o -c $(SRC)/pdlafchk.f
$(BIN)/pdgeqlrv.o:$(SRC)/pdgeqlrv.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdgeqlrv.o -c $(SRC)/pdgeqlrv.f
$(BIN)/pdgerqrv.o:$(SRC)/pdgerqrv.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdgerqrv.o -c $(SRC)/pdgerqrv.f
$(BIN)/pdtzrzrv.o:$(SRC)/pdtzrzrv.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdtzrzrv.o -c $(SRC)/pdtzrzrv.f
$(BIN)/pdgelqrv.o:$(SRC)/pdgelqrv.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdgelqrv.o -c $(SRC)/pdgelqrv.f
$(BIN)/pmatgeninc.o:$(SRC)/pmatgeninc.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pmatgeninc.o -c $(SRC)/pmatgeninc.f
$(BIN)/pdqrdriver.o:$(SRC)/pdqrdriver.f Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdqrdriver.o -c $(SRC)/pdqrdriver.f

# Link
driver:build_dirs $(BIN)/pdqrinfo.o $(BIN)/pdmatgen.o $(BIN)/pdgeqrrv.o $(BIN)/pdlafchk.o $(BIN)/pdgeqlrv.o $(BIN)/pdgerqrv.o $(BIN)/pdtzrzrv.o $(BIN)/pdgelqrv.o $(BIN)/pmatgeninc.o $(BIN)/pdqrdriver.o Makefile
	$(FC) $(CFLAGS) -o $(BIN)/pdqrdriver $(LIBS) $(BIN)/pdqrinfo.o $(BIN)/pdmatgen.o $(BIN)/pdgeqrrv.o $(BIN)/pdlafchk.o $(BIN)/pdgeqlrv.o $(BIN)/pdgerqrv.o $(BIN)/pdtzrzrv.o $(BIN)/pdgelqrv.o $(BIN)/pmatgeninc.o $(BIN)/pdqrdriver.o

clean:
	rm -f $(BIN)/pdqrdriver $(BIN)/pdqrinfo.o $(BIN)/pdmatgen.o $(BIN)/pdgeqrrv.o $(BIN)/pdlafchk.o $(BIN)/pdgeqlrv.o $(BIN)/pdgerqrv.o $(BIN)/pdtzrzrv.o $(BIN)/pdgelqrv.o $(BIN)/pmatgeninc.o $(BIN)/pdqrdriver.o

cleanall:
	rm -fr bin/*/*.o bin/*/pdqrdriver exp/*/*

