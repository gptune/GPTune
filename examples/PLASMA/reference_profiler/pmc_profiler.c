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

#include "pmc_profiler.h"

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    int ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    return ret;
}

void start_onetime_profiler()
{
    //fprintf(stderr, "start_onetime_profiler getppid: %d\n", getpid());

    if ((cpu_cycles_shmid = shmget(1234567+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((cpu_cycles = shmat(cpu_cycles_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = 4;
    pe.size = sizeof(pe);
    pe.config = 5439548;
    //static int cpu_cycles_init = 0;
    int cpu_cycles_init = 0;
    if (cpu_cycles_init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            cpu_cycles_fd[i] = perf_event_open(&pe, -1, i, -1, 0);
            if (cpu_cycles_fd[i] == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe.config);
                exit(EXIT_FAILURE);
            }
            ioctl(cpu_cycles_fd[i], PERF_EVENT_IOC_ENABLE);
            ioctl(cpu_cycles_fd[i], PERF_EVENT_IOC_RESET, 0);
        }
        cpu_cycles_init = 1;
    }

    if ((cpu_instructions_shmid = shmget(1234568+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((cpu_instructions = shmat(cpu_instructions_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    struct perf_event_attr pe2;
    memset(&pe2, 0, sizeof(pe2));
    pe2.type = PERF_TYPE_HARDWARE;
    pe2.size = sizeof(pe2);
    pe2.config = PERF_COUNT_HW_INSTRUCTIONS;
    //pe2.type = 4; //PERF_TYPE_HARDWARE;
    //pe2.size = sizeof(pe2);
    //pe2.config = 5440192; // X87 instructions

    //static int cpu_instructions_init = 0;
    int cpu_instructions_init = 0;
    if (cpu_instructions_init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            cpu_instructions_fd[i] = perf_event_open(&pe2, -1, i, -1, 0);
            if (cpu_instructions_fd[i] == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe2.config);
                exit(EXIT_FAILURE);
            }
            ioctl(cpu_instructions_fd[i], PERF_EVENT_IOC_ENABLE);
            ioctl(cpu_instructions_fd[i], PERF_EVENT_IOC_RESET, 0);
        }
        cpu_instructions_init = 1;
    }

    // RS stalls
    if ((rs_stalls_shmid = shmget(1234569+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((rs_stalls = shmat(rs_stalls_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    struct perf_event_attr pe3;
    memset(&pe3, 0, sizeof(pe3));
    pe3.type = 4;
    pe3.size = sizeof(pe3);
    pe3.config = 5439906;
    //static int rs_stalls_init = 0;
    int rs_stalls_init = 0;
    if (rs_stalls_init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            rs_stalls_fd[i] = perf_event_open(&pe3, -1, i, -1, 0);
            if (rs_stalls_fd[i] == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe3.config);
                exit(EXIT_FAILURE);
            }
            ioctl(rs_stalls_fd[i], PERF_EVENT_IOC_ENABLE);
            ioctl(rs_stalls_fd[i], PERF_EVENT_IOC_RESET, 0);
        }
        rs_stalls_init = 1;
    }

    // Fetch stalls
    if ((fetch_stalls_shmid = shmget(1234570+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((fetch_stalls = shmat(fetch_stalls_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    struct perf_event_attr pe4;
    memset(&pe4, 0, sizeof(pe4));
    pe4.type = 4;
    pe4.size = sizeof(pe4);
    pe4.config = 5440640;
    //static int fetch_stalls_init = 0;
    int fetch_stalls_init = 0;
    if (fetch_stalls_init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            fetch_stalls_fd[i] = perf_event_open(&pe4, -1, i, -1, 0);
            if (fetch_stalls_fd[i] == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe4.config);
                exit(EXIT_FAILURE);
            }
            ioctl(fetch_stalls_fd[i], PERF_EVENT_IOC_ENABLE);
            ioctl(fetch_stalls_fd[i], PERF_EVENT_IOC_RESET, 0);
        }
        fetch_stalls_init = 1;
    }

    if ((x87_instructions_shmid = shmget(1234568+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((x87_instructions = shmat(x87_instructions_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    struct perf_event_attr pe5;
    memset(&pe5, 0, sizeof(pe5));
    pe5.type = 4; //PERF_TYPE_HARDWARE;
    pe5.size = sizeof(pe5);
    pe5.config = 5440192; // X87 instructions

    //static int x87_instructions_init = 0;
    int x87_instructions_init = 0;
    if (x87_instructions_init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            x87_instructions_fd[i] = perf_event_open(&pe5, -1, i, -1, 0);
            if (x87_instructions_fd[i] == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe5.config);
                exit(EXIT_FAILURE);
            }
            ioctl(x87_instructions_fd[i], PERF_EVENT_IOC_ENABLE);
            ioctl(x87_instructions_fd[i], PERF_EVENT_IOC_RESET, 0);
        }
        x87_instructions_init = 1;
    }

    gettimeofday(&time_record1, NULL);
}

void stop_onetime_profiler()
{
    static int num_evals = 1;

    long long cpu_cycles_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(cpu_cycles_fd[i], PERF_EVENT_IOC_DISABLE, 0);
        read(cpu_cycles_fd[i], &val, sizeof(val));
        cpu_cycles_val += (val);
    }

    long long cpu_instructions_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(cpu_instructions_fd[i], PERF_EVENT_IOC_DISABLE, 0);
        read(cpu_instructions_fd[i], &val, sizeof(val));
        cpu_instructions_val += (val);
    }

    long long rs_stalls_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(rs_stalls_fd[i], PERF_EVENT_IOC_DISABLE, 0);
        read(rs_stalls_fd[i], &val, sizeof(val));
        rs_stalls_val += (val);
    }

    long long fetch_stalls_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(fetch_stalls_fd[i], PERF_EVENT_IOC_DISABLE, 0);
        read(fetch_stalls_fd[i], &val, sizeof(val));
        fetch_stalls_val += (val);
    }

    long long x87_instructions_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(x87_instructions_fd[i], PERF_EVENT_IOC_DISABLE, 0);
        read(x87_instructions_fd[i], &val, sizeof(val));
        x87_instructions_val += (val);
    }

    fprintf(stderr, "num_evals_%d_cpu_cycles: %lld\n", num_evals, cpu_cycles_val);
    fprintf(stderr, "num_evals_%d_cpu_instructions: %lld\n", num_evals, cpu_instructions_val);
    fprintf(stderr, "num_evals_%d_x87_instructions: %lld\n", num_evals, x87_instructions_val);
    fprintf(stderr, "num_evals_%d_rs_stalls: %lld\n", num_evals, rs_stalls_val);
    fprintf(stderr, "num_evals_%d_fetch_stalls: %lld\n", num_evals, fetch_stalls_val);

    *cpu_cycles = (int)cpu_cycles_val;
    *cpu_instructions = (int)cpu_instructions_val;
    *x87_instructions = (int)x87_instructions_val;
    *rs_stalls = (int)rs_stalls_val;
    *fetch_stalls = (int)fetch_stalls_val;

    gettimeofday(&time_record2, NULL);

    long long elapsed_time = (time_record2.tv_sec*1000000+time_record2.tv_usec)-(time_record1.tv_sec*1000000+time_record1.tv_usec);
    fprintf(stderr, "num_evals_%d_elapsed_time: %ld\n", num_evals, elapsed_time);
    fprintf(stderr, "num_evals_%d_IPS: %lf\n", num_evals, (float)(cpu_instructions_val)/elapsed_time)/1000000;
    fprintf(stderr, "num_evals_%d_x87_IPS: %lf\n", num_evals, (float)(x87_instructions_val)/elapsed_time)/1000000;

    for (int i = 0; i < NUM_CORES; i++) {
        close(cpu_cycles_fd[i]);
        close(cpu_instructions_fd[i]);
        close(x87_instructions_fd[i]);
        close(rs_stalls_fd[i]);
        close(fetch_stalls_fd[i]);
    }

    shmdt(cpu_cycles_shmid);
    shmctl(cpu_cycles_shmid, IPC_RMID, NULL);

    shmdt(cpu_instructions_shmid);
    shmctl(cpu_instructions_shmid, IPC_RMID, NULL);

    shmdt(rs_stalls_shmid);
    shmctl(rs_stalls_shmid, IPC_RMID, NULL);

    shmdt(fetch_stalls_shmid);
    shmctl(fetch_stalls_shmid, IPC_RMID, NULL);

    shmdt(x87_instructions_shmid);
    shmctl(x87_instructions_shmid, IPC_RMID, NULL);

    num_evals += 1;
}

int plasma_dgeqrf(int M,
        int N,
        double* A,
        int LDA,
        double* T)
{
    static int (*plasma_dgeqrf_p) (int M,
            int N,
            double* A,
            int LDA,
            double* T);
    char* error;

    start_onetime_profiler();

    if (!plasma_dgeqrf_p) {
        plasma_dgeqrf_p = dlsym(RTLD_NEXT, "plasma_dgeqrf");
        if ((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

    plasma_dgeqrf_p(M, N, A, LDA, T);

    stop_onetime_profiler();
}

int plasma_dgels(int trans,
        int M,
        int N,
        int NRHS,
        double* A,
        int LDA,
        double* T,
        double* B,
        int LDB)
{
    static int (*plasma_dgels_p) (int trans,
        int M,
        int N,
        int NRHS,
        double* A,
        int LDA,
        double* T,
        double* B,
        int LDB);
    char* error;

    start_onetime_profiler();

    if (!plasma_dgels_p) {
        plasma_dgels_p = dlsym(RTLD_NEXT, "plasma_dgels");
        if ((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

    plasma_dgels_p(trans, M, N, NRHS, A, LDA, T, B, LDB);

    stop_onetime_profiler();
}

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
        int LDC)
{
    static int (*plasma_dgemm_p) (int transA,
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
    char* error;

    start_onetime_profiler();

    if (!plasma_dgemm_p) {
        plasma_dgemm_p = dlsym(RTLD_NEXT, "plasma_dgemm");
        if ((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

    plasma_dgemm_p(transA, transB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);

    stop_onetime_profiler();
}

int plasma_dgeadd(int transa,
        int m,
        int n,
        double alpha,
        double* pA,
        int lda,
        double beta,
        double* pB,
        int ldb)
{
    static int (*plasma_dgeadd_p)(int transa,
        int m,
        int n,
        double alpha,
        double* pA,
        int lda,
        double beta,
        double* pB,
        int ldb);
    char* error;

    start_onetime_profiler();

    if (!plasma_dgeadd_p) {
        plasma_dgeadd_p = dlsym(RTLD_NEXT, "plasma_dgeadd");
        if ((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

    plasma_dgeadd_p(transa, m, n, alpha, pA, lda, beta, pB, ldb);

    stop_onetime_profiler();
}

