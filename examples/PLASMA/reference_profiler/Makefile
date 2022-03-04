CC = gcc
CCFLAGS = -O0 -DRUNTIME -shared -fPIC
LINK = -ldl -fopenmp -lpthread -lnuma -lstdc++

all: pmc_sampler pmc_profiler

pmc_sampler:
	$(CC) $(CCFLAGS) -o libpmcsampler.so pmc_sampler.c pmc_sampler.h $(LINK)

pmc_profiler:
	$(CC) $(CCFLAGS) -o libpmcprofiler.so pmc_profiler.c pmc_profiler.h $(LINK)

clean:
	rm *.so *.o *.gch
