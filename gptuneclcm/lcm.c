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

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "omp.h"
#include "mpi.h"

#include "lcm.h"

//#define DEBUG

// Macros

#define MIN(x,y) ((x<y)?(x):(y))
#define MAX(x,y) ((x>y)?(x):(y))
#define LOG_2_PI 1.8378770664093453

// Constants

char   uplo    = 'U' ;
char   trans   = 'N' ;
char   layout  = 'R' ;
int    i_zero  =  0  ;
int    i_one   =  1  ;
double d_mhalf = -0.5;
double d_half  =  0.5;
double d_one   =  1.0;

// Routines

void K
(
    // Dimensions / Sizes
    const int DI,  // dimension of tuning parameter space
    const int NT,  // #of tasks
    const int NL,  // #of latent functions in LCM
    const int m,   // #row dimension of K
    const int n,   // #column dimension of K 
    // Input arrays
    const double* restrict const theta, // size DI*NL, length scales for each Gaussian kernel k_q 
    const double* restrict const var,  // size NL, variance for each Gaussian kernel k_q 
    const double* restrict const BS,   // parameter matrices of size NT*NT of for each kernel k_q
    const double*          const X1,   // YL: This function will only be called by GPy, which seems to assume column major for X1 and X2 once inside K
    const double*          const X2,
    // Output array
    double* restrict C      // this is required by python, so C is row major
)
{
    int i, j, d, q, idxi, idxj;
    double *dists, sum;
# pragma omp parallel private ( i, j, d, q, idxi, idxj, sum, dists ) shared ( C )
    {
        dists = (double *) malloc(DI * sizeof(double));

# pragma omp for collapse(2)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                // idxi = (int) X1[i * (DI+1) + DI];  
                // idxj = (int) X2[j * (DI+1) + DI];   

                idxi = (int) X1[m * DI + i];
                idxj = (int) X2[n * DI + j];

                C[i*n + j] = 0.;

                for (d = 0; d < DI; d++)
                {
                    // dists[d] = fabs(X1[i * (DI+1) + d] - X2[j * (DI+1) + d]);
                    dists[d] = fabs(X1[m * d + i] - X2[n * d + j]);

                    dists[d] = dists[d] * dists[d];
                }
                for (q = 0; q < NL; q++)
                {
                    sum = 0.;
                    for (d = 0; d < DI; d++)
                    {
                        sum += dists[d] / theta[q * DI + d];
                    }
                    C[i*n + j] += BS[((q) * NT + idxi) * NT + idxj] * var[q] * exp( - sum);
                }
            }
        }

        free(dists);
    }

    return;
}

void rl2g(fun_jac_struct* z, int li, int prowid, int* gi)
{
    *gi = z->mb * z->nprow * (li / z->mb) + z->mb * prowid + (li % z->mb);
#ifdef DEBUG2
printf("%s %d: li %d gi %d\n", __FILE__, __LINE__, li, *gi); fflush(stdout);
#endif
}

void rg2l(fun_jac_struct* z, int gi, int* li, int* prowid)
{
    *li  = (gi / (z->mb * z->nprow)) * z->mb;
    gi = gi % (z->mb * z->nprow);
    *li += gi % z->mb;
    *prowid = gi / z->mb;
#ifdef DEBUG2
printf("%s %d: gi %d li %d prowid %d\n", __FILE__, __LINE__, gi, *li, *prowid); fflush(stdout);
#endif
}

void cl2g(fun_jac_struct* z, int lj, int pcolid, int* gj)
{
    *gj = z->mb * z->npcol * (lj / z->mb) + z->mb * pcolid + (lj % z->mb);
#ifdef DEBUG2
printf("%s %d: lj %d gj %d\n", __FILE__, __LINE__, lj, *gj); fflush(stdout);
#endif
}

void cg2l(fun_jac_struct* z, int gj, int* lj, int* pcolid)
{
    *lj  = (gj / (z->mb * z->npcol)) * z->mb;
    gj = gj % (z->mb * z->npcol);
    *lj  += gj % z->mb;
    *pcolid = gj / z->mb;
#ifdef DEBUG2
printf("%s %d: gj %d lj %d pcolid %d\n", __FILE__, __LINE__, gj, *lj, *pcolid); fflush(stdout);
#endif
}

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
    int nprow,
    int npcol,
    MPI_Comm comm
)
{
    int li, gi, lj, gj, d, idx, tid, nth, info;
    double delta;

    // fun_jac_struct structure
    fun_jac_struct* z = (fun_jac_struct *) malloc(sizeof(fun_jac_struct));

    z->DI     = DI;
    z->NT     = NT;
    z->NL     = NL;
    z->nparam = z->NL * z->DI + z->NL + z->NL * z->NT + z->NT + z->NL * z->NT;
    z->m      = m;
    z->X      = X;
    z->Y      = Y;

    // MPI ScaLAPACK related parameters
    z->mpi_comm = comm;
    MPI_Comm_rank (z->mpi_comm, &(z->pid));  /* get current process id */
    int comm_size;
    MPI_Comm_size (z->mpi_comm, &comm_size);    /* get number of processes */
#ifdef DEBUG
printf("%s %d: %d %d\n", __FILE__, __LINE__, z->pid, comm_size); fflush(stdout); MPI_Barrier( z->mpi_comm );
#endif
//    blacs_get( &i_zero, &i_zero, &(z->context) );
    z->context = Csys2blacs_handle(z->mpi_comm);
#ifdef DEBUG
printf("%s %d: %d\n", __FILE__, __LINE__, z->context); fflush(stdout); MPI_Barrier( z->mpi_comm );
#endif
   Cblacs_gridinit( &(z->context), &layout, nprow, npcol);
    // blacs_gridinit_( &(z->context), &layout, &nprow, &npcol);
#ifdef DEBUG
printf("%s %d: %d %d\n", __FILE__, __LINE__, nprow, npcol); fflush(stdout); MPI_Barrier( z->mpi_comm );
#endif
//    int usermap[comm_size];
//    MPI_Bcast( usermap, comm_size, MPI_DOUBLE, z->alphadesc[6]*z->npcol + z->alphadesc[7], z->mpi_comm );
//    MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm z->mpi_comm)
//    blacs_gridmap(&(z->context), usermap, &npcol, &nprow, &npcol);
//printf("@@@@@ %d %d %d %d %d %d\n", z->pid, z->context, (z->nprow), (z->npcol), (z->prowid), (z->pcolid)); fflush(stdout);
    Cblacs_gridinfo( (z->context), &(z->nprow), &(z->npcol), &(z->prowid), &(z->pcolid) );
	// blacs_gridinfo_( &(z->context), &(z->nprow), &(z->npcol), &(z->prowid), &(z->pcolid) );
#ifdef DEBUG
printf("%s %d: %d %d %d %d\n", __FILE__, __LINE__, (z->nprow), (z->npcol), (z->prowid), (z->pcolid)); fflush(stdout); MPI_Barrier( z->mpi_comm );
#endif
//printf("##### %d %d %d %d %d %d\n", z->pid, z->context, (z->nprow), (z->npcol), (z->prowid), (z->pcolid)); fflush(stdout);
    
    z->mb = mb;
    // z->lr = numroc_( &m, &mb, &(z->prowid), &i_zero, &nprow );
    // z->lc = numroc_( &m, &mb, &(z->pcolid), &i_zero, &npcol );
	
	z->lr = PB_Cnumroc( m, 0, mb, mb, (z->prowid), i_zero, nprow );
	z->lc = PB_Cnumroc( m, 0, mb, mb, (z->pcolid), i_zero, npcol );
#ifdef DEBUG
printf("%s %d: %d %d\n", __FILE__, __LINE__, z->lr, z->lc); fflush(stdout); MPI_Barrier( comm );
#endif

	if(z->prowid!=-1 && z->pcolid!=-1){
		int lr_tmp = MAX(z->lr,1);
		descinit_ (&(z->Kdesc), &m, &m, &mb, &mb, &i_zero, &i_zero, &(z->context), &lr_tmp, &info);
#ifdef DEBUG
printf("%s %d: Kdesc %d %d %d %d %d %d %d %d %d \n", __FILE__, __LINE__, z->Kdesc[0], z->Kdesc[1],z->Kdesc[2], z->Kdesc[3], z->Kdesc[4], z->Kdesc[5], z->Kdesc[6], z->Kdesc[7], z->Kdesc[8]); fflush(stdout); MPI_Barrier( comm );
#endif
		descinit_ (&(z->alphadesc), &m, &i_one, &mb, &i_one, &i_zero, &i_zero, &(z->context), &lr_tmp, &info);
#ifdef DEBUG
printf("%s %d: alphadesc %d %d %d %d %d %d %d %d %d\n", __FILE__, __LINE__, z->alphadesc[0], z->alphadesc[1],z->alphadesc[2], z->alphadesc[3], z->alphadesc[4], z->alphadesc[5], z->alphadesc[6], z->alphadesc[7], z->alphadesc[8]); fflush(stdout); MPI_Barrier( comm );
#endif
	}else{
		z->Kdesc[1]=-1;
		z->alphadesc[1]=-1;
	} 
	
    // Allocate shared arrays
 
    z->dists  = (double *) malloc(z->lr * z->lc * DI * sizeof(double));
    z->exps   = (double *) malloc(z->lr * z->lc * NL * sizeof(double));
    z->alpha  = (double *) malloc(z->lr              * sizeof(double));
    z->distY  = (double *) malloc(z->lr              * sizeof(double));
    z->K      = (double *) malloc(z->lr * z->lc      * sizeof(double));
    z->buffer = (double *) malloc(z->nparam          * sizeof(double));

    nth = omp_get_max_threads();
    z->gradients_TPS = (double **) malloc(nth * sizeof(double*));

# pragma omp parallel private ( li, gi, lj, gj, d, idx, tid, delta ) shared ( z )
    {
        // Allocate private arrays

        tid = omp_get_thread_num();
        z->gradients_TPS[tid] = (double *) calloc(z->nparam, sizeof(double));

//# pragma omp for collapse(3)
# pragma omp for
        for (li = 0; li < z->lr; li++)
        {
            // Compute element-wise square distances
            rl2g(z, li, z->prowid, &gi);
            for (lj = 0; lj < z->lc; lj++)
            {
                cl2g(z, lj, z->pcolid, &gj);
                for (d = 0; d < DI; d++)
                {
                    idx = (li * z->lc + lj) * DI + d;
                    delta = X[gi * (DI + 1) + d] - X[gj * (DI + 1) + d];
                    z->dists[idx] = delta * delta;
                }
            }
            // Copy Y in distY
            z->distY[li] = z->Y[gi];
        }
    }

    return z;
}

void finalize
(
    // fun_jac_struct structure
    fun_jac_struct* z 
)
{
# pragma omp parallel shared ( z )
    {
        // Deallocate private arrays

        int tid = omp_get_thread_num();
        free(z->gradients_TPS[tid]);
    }

    // Deallocate shared arrays

    free(z->gradients_TPS);

    free(z->dists);
    free(z->exps);
    free(z->alpha);
    free(z->distY);
    free(z->K);
    free(z->buffer);

    
	if(z->context!=-1){
	// blacs_gridexit_( &(z->context) );
    Cblacs_gridexit( z->context );
	}
//    int tmp = 1; // 1 instead of 0, as MPI might be called after blacs exits
//    blacs_exit( &tmp );

    free(z);
}

double fun_jac // negloglike_and_grads
(
    // Input parameters
    double* params,
    // fun_jac_struct structure
    fun_jac_struct* z,
    // Output gradients
    double* gradients
)
{
    // Declare variables

    int k, li, gi, lj, ljstart, gj, d, q, idxi, idxj, idxk, info, tmppid;
    double sum, ws2, a, dldk, *dL_dK, t1, t2;

    // Unpack hyper-parameters

    double* theta = params;                 // length scales of each kernel k_q
    double* var   = theta + z->NL * z->DI;  // variance of each kernel k_q
    double* kappa = var   + z->NL;          // YL: diagonal regularizer in B_q ??? not documented in the paper?  
    double* sigma = kappa + z->NL * z->NT;  // diagonal matrix D of variances in LCM
    double* ws    = sigma + z->NT;          // W_q used to form B_q
    
    // Initialize outputs

    double neg_log_marginal_likelihood = 0.;

    t1 = omp_get_wtime();

    for (k = 0; k < z->nparam ; k++)
    {
        z->buffer[k] = 0.;
    }

# pragma omp parallel private ( k, li, gi, lj, ljstart, gj, d, q, idxi, idxj, idxk, sum, info, tmppid ) shared ( z, theta, var, kappa, sigma, ws )
    {
        int tid = omp_get_thread_num();

        // Initialize private arrays

        for (k = 0; k < z->nparam; k++)
        {
            z->gradients_TPS[tid][k] = 0.;
        }

# pragma omp for
        for (k = 0; k < z->lr * z->lc; k++)
        {
            z->K[k] = 0.;
        }

# pragma omp for
        for (li = 0; li < z->lr; li++)
        {
            rl2g(z, li, z->prowid, &gi);
            idxi = (int) z->X[gi * (z->DI + 1) + z->DI];

#ifdef DEBUG
printf("%s %d: %d %d %d %d\n", __FILE__, __LINE__, z->pid, li, gi, idxi); fflush(stdout);
#endif

            for (lj = 0; lj < z->lc; lj++)
            {
                cl2g(z, lj, z->pcolid, &gj);
                idxj = (int) z->X[gj * (z->DI + 1) + z->DI];
				if(gi<=gj){  //only store the upper triangular part 
	#ifdef DEBUG
	printf("%s %d: %d %d %d %d\n", __FILE__, __LINE__, z->pid, lj, gj, idxj); fflush(stdout);
	#endif
	//@                idxk =li * z->lc + lj;
					idxk = lj * z->lr + li;              //K is needed for ScaLAPACK, so column major

					z->K[idxk] = 0.;

					for (q = 0; q < z->NL; q++)
					{
						sum = 0.;
						for (d = 0; d < z->DI; d++)
						{
							sum += z->dists[(li * z->lc + lj) * z->DI + d] / theta[q * z->DI + d];
						}
						z->exps[(li * z->lc + lj) * z->NL + q] = exp( - sum );
						if (idxi == idxj)
						{
							z->K[idxk] += (ws[q * z->NT + idxi] * ws[q * z->NT + idxj] + kappa[q * z->NT + idxi]) * var[q] * z->exps[(li * z->lc + lj) * z->NL + q];
						}
						else
						{
							z->K[idxk] += ws[q * z->NT + idxi] * ws[q * z->NT + idxj] * var[q] * z->exps[(li * z->lc + lj) * z->NL + q]; //dsyrk and dgbmv
						}
					}
				}
                // printf("nima %5d%5d\n",idxi,idxj);
            }
			
            cg2l(z, gi, &ljstart, &tmppid);			
            if (z->pcolid == tmppid)
            {
//@                z->K[li * z->lc + ljstart] += sigma[idxi] + 1e-8;
                z->K[ljstart * z->lr + li] += sigma[idxi] + 1e-8;     //YL: why is 1e-8 here?
                // printf("%5d%5d%14f\n",ljstart,li,z->K[ljstart * z->lr + li]);
            }
        }
    }

#ifdef DEBUG
for (int p = 0; p < 8; p++)
{
    if (z->pid == p)
    {
        printf("m%d = [", p);
        for (int lj = 0; lj < z->lc; lj++)
        {
            printf("[%e", z->K[lj * z->lr]);
            for (int li = 1; li < z->lr; li++)
            {
                printf(", %e", z->K[lj * z->lr + li]);
            }
            printf("], ");
        }
        printf("]\n");
        fflush(stdout);
    }
    MPI_Barrier( z->mpi_comm );
}
//exit(0);
#endif
    /**************************************************************************************************/

    // Compute dL_dK
	if(z->prowid!=-1 && z->pcolid!=-1){
		pdpotrf_( &uplo, &(z->m), z->K, &i_one, &i_one, &(z->Kdesc), &info );
	}
/*    if (info != 0)
    {
        return INFINITY;
    }

    MPI_Barrier( z->mpi_comm );
    printf("pid %d info %d", z->pid, info);
*/
#ifdef DEBUG
for (int p = 0; p < 8; p++)
{
    if (z->pid == p)
    {
        printf("m%d = [", p);
        for (int lj = 0; lj < z->lc; lj++)
        {
            printf("[%e", z->K[lj * z->lr]);
            for (int li = 1; li < z->lr; li++)
            {
                printf(", %e", z->K[lj * z->lr + li]);
            }
            printf("], ");
        }
        printf("]\n");
        fflush(stdout);
    }
    MPI_Barrier( z->mpi_comm );
}
//exit(0);
#endif

    double W_logdet = 0., W_logdet2 = 0.;
    for (li = 0; li < z->lr; li++)
    {
        rl2g(z, li, z->prowid, &gi);
        cg2l(z, gi, &lj, &tmppid);
        if (z->pcolid == tmppid)
        {
//@            W_logdet2 += log(z->K[li * z->lc + lj]);
            W_logdet2 += log(z->K[lj * z->lr + li]);
        }
    }
    W_logdet2 *= 2.;
//printf("!!!!! %d %f\n", z->pid, W_logdet2); fflush(stdout);
    int comm_size;
    MPI_Comm_size (z->mpi_comm, &comm_size);    /* get number of processes */
    // printf("!!! %d\n", comm_size);
    // printf("!!! %d\n", comm_size);
    MPI_Allreduce( &W_logdet2, &W_logdet, 1, MPI_DOUBLE, MPI_SUM, z->mpi_comm);
    // printf("!!!\n");

    // Copy Y in alpha as dpotrs computes the solution of A x = b in place
    for (li = 0; li < z->lr; li++)
    {
        z->alpha[li] = z->distY[li];
    }
	
	if(z->prowid!=-1 && z->pcolid!=-1){
		pdpotrs_( &uplo, &(z->m), &i_one, z->K, &i_one, &i_one, &(z->Kdesc), z->alpha, &i_one, &i_one, &(z->alphadesc), &info );

		pdpotri_( &uplo, &(z->m), z->K, &i_one, &i_one, &(z->Kdesc), &info );
	}
//YL: check https://gpy.readthedocs.io/en/deploy/GPy.likelihoods.html for the gradient computation	
	
	
	
    double dot = 0.;
	if(z->prowid!=-1 && z->pcolid!=-1){	
		pddot_ (&(z->m), &dot, z->alpha, &i_one, &i_one, z->alphadesc, &i_one, z->distY,  &i_one, &i_one, z->alphadesc, &i_one);
	}	
    MPI_Bcast( &dot, 1, MPI_DOUBLE, z->alphadesc[6]*z->npcol + z->alphadesc[7], z->mpi_comm );
//printf("@@@@@ %d %f\n", z->pid, dot); fflush(stdout);

    neg_log_marginal_likelihood = 0.5 * (z->m * LOG_2_PI + W_logdet + dot);

    dL_dK = z->K;
	if(z->prowid!=-1 && z->pcolid!=-1){
		pdsyrk_( &uplo, &trans, &(z->m), &i_one, &d_half, z->alpha, &i_one, &i_one, z->alphadesc, &d_mhalf, dL_dK, &i_one, &i_one, z->Kdesc);
	}
    /**************************************************************************************************/

# pragma omp parallel private ( k, li, gi, lj, ljstart, gj, d, q, idxi, idxj, idxk, sum, ws2, a, dldk, info, tmppid ) shared ( z, theta, var, kappa, sigma, ws )
    {
        int tid = omp_get_thread_num();

        // Unpack gradients_TPS

        double* theta_gradients_TPS = z->gradients_TPS[tid];
        double* var_gradients_TPS   = theta_gradients_TPS + z->NL * z->DI;
        double* kappa_gradients_TPS = var_gradients_TPS   + z->NL;
        double* sigma_gradients_TPS = kappa_gradients_TPS + z->NL * z->NT;
        double* ws_gradients_TPS    = sigma_gradients_TPS + z->NT;

        // Compute gradients

# pragma omp for
        for (li = 0; li < z->lr; li++)
        {

            rl2g(z, li, z->prowid, &gi);
            idxi = (int) z->X[gi * (z->DI + 1) + z->DI];

            cg2l(z, gi, &ljstart, &tmppid);
            if (z->pcolid == tmppid) // Diagonal elements
            {
//@                idxk = li * z->lc + ljstart;
                idxk = ljstart * z->lr + li;
                dldk = dL_dK[idxk];

                sigma_gradients_TPS[idxi] += dldk;

                for (q = 0; q < z->NL; q++)
                {
                    kappa_gradients_TPS[q * z->NT + idxi] += dldk;
                }

                for (q = 0; q < z->NL; q++)
                {
                    ws2 = ws[q * z->NT + idxi] * ws[q * z->NT + idxi];
                    a = dldk * z->exps[(li * z->lc + ljstart) * z->NL + q];
                    var_gradients_TPS[q] += ws2 * a;
                    a *= var[q];
                    for (d = 0; d < z->DI; d++)
                    {
                        theta_gradients_TPS[q * z->DI + d] += ws2 * a * (z->dists[(li * z->lc + ljstart) * z->DI + d]) / (theta[q * z->DI + d] * theta[q * z->DI + d]);
                    }
                    // If (idxi == idxj) then ws_gradient is supposed to be 2 * ws[] * a
                    // which is exacly what happens in the following two lines anyways
                    // so no need for an if statement
                    ws_gradients_TPS[q * z->NT + idxi] += ws[q * z->NT + idxi] * a;
                    ws_gradients_TPS[q * z->NT + idxi] += ws[q * z->NT + idxi] * a;
                }
            }

            for (lj = 0; lj < z->lc; lj++)
            {
                cl2g(z, lj, z->pcolid, &gj);
                idxj = (int) z->X[gj * (z->DI + 1) + z->DI];
				if (gi <= gj){		
					
	//@                idxk = li * z->lc + lj;
					idxk = lj * z->lr + li;
					dldk = dL_dK[idxk];

					if (idxi == idxj)
					{
						for (q = 0; q < z->NL; q++)
						{
							kappa_gradients_TPS[q * z->NT + idxi] += 2. * dldk;
						}
					}

					for (q = 0; q < z->NL; q++)
					{
						ws2 = ws[q * z->NT + idxi] * ws[q * z->NT + idxj];
						a = dldk * z->exps[(li * z->lc + lj) * z->NL + q];
						var_gradients_TPS[q] += 2. * ws2 * a;
						a *= var[q];
						for (d = 0; d < z->DI; d++)
						{
							theta_gradients_TPS[q * z->DI + d] += 2. * ws2 * a * (z->dists[(li * z->lc + lj) * z->DI + d]) / (theta[q * z->DI + d] * theta[q * z->DI + d]);
						}
						// If (idxi == idxj) then ws_gradient is supposed to be 2 * ws[] * a
						// which is exacly what happens in the following two lines anyways
						// so no need for an if statement
						ws_gradients_TPS[q * z->NT + idxi] += 2. * ws[q * z->NT + idxj] * a;
						ws_gradients_TPS[q * z->NT + idxj] += 2. * ws[q * z->NT + idxi] * a;
					}
				}	
            }
        }

        // Reduce private arrays

# pragma omp critical
        {
            for (k = 0; k < z->nparam; k++)
            {
                z->buffer[k] += z->gradients_TPS[tid][k];
            }
        }
    }

    MPI_Allreduce(z->buffer, gradients, z->nparam, MPI_DOUBLE, MPI_SUM, z->mpi_comm);
    
    t2 = omp_get_wtime();
    // if (z->pid == 0){
    //     printf("time in fun_jac: %e\n",t2-t1);
    //     fflush(stdout);
    // }
    return neg_log_marginal_likelihood;
}

//printf("~ %d\n", z->pid);
//printf("Kdesc %d %d %d %d %d %d %d %d %d\n", z->Kdesc[0], z->Kdesc[1],z->Kdesc[2], z->Kdesc[3], z->Kdesc[4], z->Kdesc[5], z->Kdesc[6], z->Kdesc[7], z->Kdesc[8]);

//fflush(stdout);

//MPI_Barrier( MPI_COMM_WORLD );
//MPI_Barrier( z->mpi_comm );

