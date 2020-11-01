// GPTune Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S.Dept. of Energy) and the University of
// California, Berkeley.  All rights reserved.
//
// If you have questions about your rights to use or distribute this software, 
// please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
//
// NOTICE. This Software was developed under funding from the U.S. Department 
// of Energy and the U.S. Government consequently retains certain rights.
// As such, the U.S. Government has been granted for itself and others acting
// on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
// the Software to reproduce, distribute copies to the public, prepare
// derivative works, and perform publicly and display publicly, and to permit
// other to do so.

//#include "mkl_blas.h"
//#include "mkl_lapack.h"

/* Interfaces */

// BLACS
void blacs_get(const int *ConTxt, const int *what, int *val);
void blacs_gridinit(int *ConTxt, const char *layout, const int *nprow, const int *npcol);
void blacs_gridmap(int *ConTxt, const int *usermap, const int *ldup, const int *nprow0, const int *npcol0);
void blacs_gridinfo(int *ConTxt, int *nprow, int *npcol, int *myprow, int *mypcol );
void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
void blacs_gridexit(const int *ConTxt);
void blacs_exit(const int *notDone);

// ScaLAPACK
int numroc(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
void pdpotrf(const char* uplo, const int* n, double* a, const int* ia, const int* ja, const int* desca, int* info);
void pdgemr2d (int *m , int *n , double *a , int *ia , int *ja , int *desca , double *b , int *ib , int *jb , int *descb , int *ictxt );
void pdpotrs(const char* uplo, const int* n, const int* nrhs, const double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, int* info);
void pdpotri(const char* uplo, const int* n, double* a, const int* ia, const int* ja, const int* desca, int* info);

// PBLAS
void pddot_(int * N, double * DOT, double * X, int * IX, int * JX, int * DESCX, int * INCX, double * Y, int * IY, int * JY, int * DESCY, int * INCY);
void pdsyrk_(char* UPLO, char* TRANS, int * N, int * K, double * ALPHA, double * A, int * IA, int * JA, int * DESCA, double * BETA, double * C, int * IC, int * JC, int * DESCC);

/* LCM structure */

typedef struct
{
    /* Sequential */

    // Input dimensions and sizes

    int DI;
    int NT;
    int NL;
    int nparam;
    int m;

    // Input arrays

    //// Data

    double* X;
    double* Y;

    // Work arrays

    double*  dists;
    double*  exps;
    double*  alpha;
    double*  K;

    /* OpenMP */

    // Work arrays

    double** gradients_TPS;

    /* MPI & ScaLAPACK */

    int mb;            // Blocking factor used to distribute the rows of the global matrix K in ScaLAPACK
    int lr;            // Local number of rows of K matrix
    int lc;            // Local number of columns of K matrix
    int maxtries;      // Max number of jittering 
    int nprow;         // Number of rows process
    int npcol;         // Number of columns process
    int pid;           // Process ID in communicator mpi_comm
    int prowid;        // Process row ID in communicator mpi_comm
    int pcolid;        // Process col ID in communicator mpi_comm
    int context;       // BLACS context
    int Kdesc[9];      // ScaLAPACK K matrix descriptor
    int alphadesc[9];  // ScaLAPACK alpha vector descriptor
    double* distY;     // Distributed version of the Y array
    double* buffer;    // buffer for MPI communications and for internal copies
    MPI_Comm mpi_comm; // MPI communicator

} fun_jac_struct;

/* LCM routines */

fun_jac_struct* initialize
(
    // Dimensions / Sizes
    int DI,
    int NT,
    int NL,
    int m,
    // Input array
    double* X,
    double* Y,
    // MPI ScaLAPACK related parameters
    int mb,
    int maxtries,
    int nprow,
    int npcol,
    MPI_Comm comm
);

void finalize
(
    // fun_jac_struct structure
    fun_jac_struct* z 
);

double fun_jac // negloglike_and_grads
(
    // Input parameters
    double* params,
    // fun_jac_struct structure
    fun_jac_struct* z,
    // Output gradients
    double* gradients
);

