CC = gcc
CCFLAGS = -O0 -DRUNTIME -shared -fPIC
LINK = -ldl -fopenmp -lpthread -lnuma -lstdc++
#LINK = 

all: reference_profiler

reference_profiler:
	$(CC) $(CCFLAGS) -o librefprof.so reference_profiler.c reference_profiler.h $(LINK)
clean:
	rm librefprof.so *.o *.gch
