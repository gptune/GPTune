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

#include "pmc_sampler.h"

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    int ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    return ret;
}

void start_profiler_thread()
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    //pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe.config = 5440192; // X87 instructions
    printf("pe.config: %ld\n", pe.config);

    int fd = perf_event_open(&pe, getppid(), -1, -1, 0);
    if (fd == -1) {
        fprintf(stderr, "Error opening leader %llx\n", pe.config);
        exit(EXIT_FAILURE);
    }

    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE);
    event_descriptor = fd;
}

void stop_profiler_thread()
{
    long long cpu_instructions_val = 0;
    long long val;
    ioctl(event_descriptor, PERF_EVENT_IOC_DISABLE, 0);
    read(event_descriptor, &val, sizeof(val));
    close(event_descriptor);
    cpu_instructions_val += (val);
    fprintf(stderr, "#cpu_instructions: %lld\n", cpu_instructions_val);
    *cpu_instructions = (int)cpu_instructions_val;
}

void start_profiler()
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = 4; //PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    //pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe.config = 5440192; // X87 instructions
    //printf("pe.config: %ld\n", pe.config);
    //
    static int init = 0;
    if (init == 0) {
        for (int i = 0; i < NUM_CORES; i++) {
            int fd = perf_event_open(&pe, -1, i, -1, 0);
            if (fd == -1) {
                fprintf(stderr, "Error opening leader %llx\n", pe.config);
                exit(EXIT_FAILURE);
            }
            event_descriptors[i] = fd;
        }
        init = 1;
    }

    for (int i = 0; i < NUM_CORES; i++) {
        ioctl(event_descriptors[i], PERF_EVENT_IOC_ENABLE);
        ioctl(event_descriptors[i], PERF_EVENT_IOC_RESET, 0);
    }

    //for (int i = 0; i < NUM_CORES; i++) {
    //    int fd = perf_event_open(&pe, -1, i, -1, 0);
    //    if (fd == -1) {
    //        fprintf(stderr, "Error opening leader %llx\n", pe.config);
    //        exit(EXIT_FAILURE);
    //    }

    //    //ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    //    ioctl(fd, PERF_EVENT_IOC_ENABLE);
    //    event_descriptors[i] = fd;
    //}
}

void stop_profiler()
{
    long long cpu_instructions_val = 0;
    for (int i = 0; i < NUM_CORES; i++) {
        long long val;
        ioctl(event_descriptors[i], PERF_EVENT_IOC_DISABLE, 0);
        read(event_descriptors[i], &val, sizeof(val));
        //close(event_descriptors[i]);
        cpu_instructions_val += (val);
        //fprintf(stderr, "#cpu_instructions (core: %d): %lld\n", i, val);
    }
    //fprintf(stderr, "#cpu_instructions: %lld\n", cpu_instructions_val);
    *cpu_instructions = (int)cpu_instructions_val;
}

void *perf_sampler(void *args)
{
    //if ((cpu_cycles_shmid = shmget(123456, sizeof(int), IPC_CREAT | 0666)) < 0)
    //{
    //    printf("Error getting shared memory id");
    //    exit(1);
    //}
    //if ((cpu_cycles = shmat(cpu_cycles_shmid, NULL, 0)) == (int*) -1)
    //{
    //    printf("Error attaching shared memory id");
    //    exit(1);
    //}

    fprintf(stderr, "getppid: %d", getpid());
    if ((cpu_instructions_shmid = shmget(1234567+getpid(), sizeof(int), IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        exit(1);
    }
    if ((cpu_instructions = shmat(cpu_instructions_shmid, NULL, 0)) == (int*) -1)
    {
        printf("Error attaching shared memory id");
        exit(1);
    }

    while (sampling_running) {
        start_profiler();
        usleep(1000000);
        stop_profiler();
    }

    for (int i = 0; i < NUM_CORES; i++) {
        close(event_descriptors[i]);
    }

    //shmdt(cpu_cycles_shmid);
    //shmctl(cpu_cycles_shmid, IPC_RMID, NULL);

    shmdt(cpu_instructions_shmid);
    shmctl(cpu_instructions_shmid, IPC_RMID, NULL);

    return NULL;
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

    sampling_running = 1;
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, perf_sampler, NULL);

    fprintf(stderr, "plasma_dgeqrf interpositioning\n");
    if (!plasma_dgeqrf_p) {
        plasma_dgeqrf_p = dlsym(RTLD_NEXT, "plasma_dgeqrf");
        if ((error = dlerror()) != NULL) {
            fputs(error, stderr);
            exit(1);
        }
    }

    plasma_dgeqrf_p(M, N, A, LDA, T);
    sampling_running = 0;
    pthread_join(thread_id, NULL);
}


