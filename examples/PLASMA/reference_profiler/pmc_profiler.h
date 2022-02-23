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
#include <sys/time.h>

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

void start_profiler();
void stop_profiler();
void* perf_sampler(void* args);

int cpu_cycles_shmid;
int cpu_cycles_fd[NUM_CORES];
int* cpu_cycles;

int cpu_instructions_shmid;
int cpu_instructions_fd[NUM_CORES];
int* cpu_instructions;

int x87_instructions_shmid;
int x87_instructions_fd[NUM_CORES];
int* x87_instructions;

int rs_stalls_shmid;
int rs_stalls_fd[NUM_CORES];
int* rs_stalls;

int fetch_stalls_shmid;
int fetch_stalls_fd[NUM_CORES];
int* fetch_stalls;

struct timeval time_record1;
struct timeval time_record2;

int plasma_dgeqrf(int M,
        int N,
        double* A,
        int LDA,
        double* T);

int plasma_dgels(int trans,
        int M,
        int N,
        int NRHS,
        double* A,
        int LDA,
        double* T,
        double* B,
        int LDB);

int plasma_dgemm(int transA,
        int transB,
        int M,
        int N,
        int K,
        double alpha,
        double* A,
        int LDA,
        double* B,
        int LDB,
        double beta,
        double* C,
        int LDC);

int plasma_dgeadd(int transa,
        int m,
        int n,
        double alpha,
        double* pA,
        int lda,
        double beta,
        double* pB,
        int ldb);

#endif
