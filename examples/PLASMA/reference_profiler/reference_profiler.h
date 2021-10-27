/// GPTune Copyright (c) 2019, The Regents of the University of California,
/// through Lawrence Berkeley National Laboratory (subject to receipt of any
/// required approvals from the U.S.Dept. of Energy) and the University of
/// California, Berkeley.  All rights reserved.
///
/// If you have questions about your rights to use or distribute this software,
/// please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
///
/// NOTICE. This Software was developed under funding from the U.S. Department
/// of Energy and the U.S. Government consequently retains certain rights.
/// As such, the U.S. Government has been granted for itself and others acting
/// on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
/// the Software to reproduce, distribute copies to the public, prepare
/// derivative works, and perform publicly and display publicly, and to permit
/// other to do so.

#ifndef __REFERENCE_PROFILER_H_
#define __REFERENCE_PROFILER_H_

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <dlfcn.h>

#ifndef __NR_perf_event_open
#if defined __PPC__
#define __NR_perf_event_open 319
#elif defined __i386__
#define __NR_perf_event_open 336
#elif defined __x86_64__
#define __NR_perf_event_open 298
#elif defined __tile__
#define __NR_perf_event_open 241
#error unknown target architecture
#endif
#endif

#define NUM_CORES 64

pid_t mypid;
int sampling_running = 0;
int event_descriptors[NUM_CORES];

int event_descriptor;

void start_profiler();
void stop_profiler();
void* perf_sampler(void* args);

int cpu_cycles_shmid;
int cpu_instructions_shmid;
int* cpu_cycles;
int* cpu_instructions;

int plasma_dgeqrf(int M,
        int N,
        double* A,
        int LDA,
        double* T);

#endif
