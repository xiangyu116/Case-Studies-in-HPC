#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);

/* Do QR on W (rows x n) in-place (column-major), return its R in Rout (n x n) */
static void qr_get_R(double *W, int rows, int n, double *Rout) {
    int info=0;
    int lda=rows;
    int k=(rows<n) ? rows : n;

    double *tau=(double*)malloc(sizeof(double)*k);

    // workspace query
    int lwork=-1;
    double wkopt=0.0;
    dgeqrf_(&rows, &n, W, &lda, tau, &wkopt, &lwork, &info);

    if(info!=0) 
    {
        fprintf(stderr, "dgeqrf workspace query failed info=%d\n", info);
        MPI_Abort(MPI_COMM_WORLD, info);
    }
    
    lwork=(int)wkopt+1;
    
    double *work=(double*)malloc(sizeof(double)*lwork);
  
    dgeqrf_(&rows, &n, W, &lda, tau, work, &lwork, &info);
    if (info!=0) 
    {
        fprintf(stderr, "dgeqrf failed info=%d\n", info);
        MPI_Abort(MPI_COMM_WORLD, info);
    }

    memset(Rout, 0, sizeof(double)*n*n);
    for (int j=0;j<n;++j) 
    {
        for (int i=0;i<=j && i<rows;++i) 
        {
            Rout[i+j*n]=W[i+j*rows];
        }
    } 

    free(work);
    free(tau);
}

/* Pack row-block r (rows r*mb .. (r+1)*mb-1) from A(m x n) into buf(mb x n), column-major */
static void pack_row_block(const double *A, int m, int n, int r, int mb, double *buf) {
    int row0=r * mb;
    for (int j=0;j<n;++j) 
    {
        memcpy(&buf[j*mb], &A[row0 + j*m], sizeof(double)*mb);
    }
}

/* Build stacked matrix W = [Rtop; Rbot] of size (2n x n), column-major */
static void stack_Rs(const double *Rtop, const double *Rbot, int n, double *W) {
    int rows=2*n;
    for (int j=0;j<n;++j) 
    {
        memcpy(&W[0+j*rows], &Rtop[0+j*n], sizeof(double)*n);
        memcpy(&W[n+j*rows], &Rbot[0+j*n], sizeof(double)*n);
    }
}

/* TSQR for exactly 4 ranks, returns final R on rank 0 */
static void TSQR_4(const double *A_local_in, int mb, int n, int rank, double *R_final_rank0) 
{
    // local QR -> R_local
    double *A_local=(double*)malloc(sizeof(double)*mb*n);
    double *R_local=(double*)malloc(sizeof(double)*n*n);

    memcpy(A_local, A_local_in, sizeof(double)*mb*n);
    qr_get_R(A_local, mb, n, R_local);

    if (rank==1) 
    {
        MPI_Send(R_local, n*n, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
    } 
    else if (rank==3) 
    {
        MPI_Send(R_local, n*n, MPI_DOUBLE, 2, 11, MPI_COMM_WORLD);
    } 
    else if (rank==2) 
    {
        // recv R3
        double *R3=(double*)malloc(sizeof(double)*n*n);
        MPI_Recv(R3, n*n, MPI_DOUBLE, 3, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // QR([R2; R3]) -> R23
        double *W23=(double*)malloc(sizeof(double)*(2*n)*n);
        double *R23=(double*)malloc(sizeof(double)*n*n);

        stack_Rs(R_local, R3, n, W23);
        qr_get_R(W23, 2*n, n, R23);

        // send R23 to rank0
        MPI_Send(R23, n*n, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);

        free(R23); 
        free(W23); free(R3);
    } 
    else if (rank==0) 
    {
        // recv R1
        double *R1=(double*)malloc(sizeof(double)*n*n);
        MPI_Recv(R1, n*n, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // QR([R0; R1]) -> R01
        double *W01=(double*)malloc(sizeof(double)*(2*n)*n);
        double *R01=(double*)malloc(sizeof(double)*n*n);

        stack_Rs(R_local, R1, n, W01);
        qr_get_R(W01, 2*n, n, R01);

        // recv R23 from rank2
        double *R23=(double*)malloc(sizeof(double)*n*n);
        MPI_Recv(R23, n*n, MPI_DOUBLE, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // final QR([R01; R23]) -> R
        stack_Rs(R01, R23, n, W01);
        qr_get_R(W01, 2*n, n, R_final_rank0);

        free(R23);
         free(R01); 
         free(W01); 
         free(R1);
    }

    free(R_local);
    free(A_local);
}

int main(int argc, char **argv) 
{
    MPI_Init(&argc, &argv);

    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size!=4) 
    {
        if (rank==0) 
        {
          fprintf(stderr, "Run with exactly 4 MPI processes: mpirun -np 4 ./tsqr\n");
        }
        MPI_Finalize();
        return 1;
    }


    int m=(argc > 1) ? atoi(argv[1]) : 400;   // default 400 if not provided
    int n=(argc > 2) ? atoi(argv[2]) : 20;    // default 20 if not provided
    if (m%4!=0) 
    {
      m=4*(m/4);
    }
    int mb=m/4;

    double *A=NULL;                 // full A on rank0 (for verification)
    double *sendbuf=NULL;           // packed blocks on rank0
    double *A_local=(double*)malloc(sizeof(double)*mb*n);

    if (rank==0) 
    {
        A=(double*)malloc(sizeof(double)*m*n);
        sendbuf=(double*)malloc(sizeof(double)*m*n); // 4 blocks, each mb*n

        srand(0);
        // column-major fill
        for (int j=0;j<n;++j) 
        {
            for (int i=0;i<m;++i) 
            {
                A[i+j*m]=((double)rand()/RAND_MAX)-0.5;
            }
        }

        // pack 4 blocks into sendbuf contiguous chunks
        for (int r=0;r<4;++r) 
        {
            pack_row_block(A, m, n, r, mb, &sendbuf[r*mb*n]);
        }
    }

    MPI_Scatter(rank==0 ? sendbuf : NULL, mb*n, MPI_DOUBLE, A_local, mb*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *R=NULL;
    if (rank==0) 
    {
        R=(double*)malloc(sizeof(double)*n*n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0=MPI_Wtime();

    TSQR_4(A_local, mb, n, rank, R);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1=MPI_Wtime();

    if (rank==0) 
    {
        /* third question wants scaling plots -> output m n time */
        printf("%d %d %.6e\n", m, n, t1 - t0);
        fflush(stdout);
    }

    // ---- verify on rank0: A^T A ≈ R^T R ----
    if (rank==0) 
    {
        double *AtA=(double*)malloc(sizeof(double)*n*n);
        double *RtR=(double*)malloc(sizeof(double)*n*n);

        memset(AtA, 0, sizeof(double)*n*n);
        memset(RtR, 0, sizeof(double)*n*n);
        // AtA
        for(int j=0;j<n;++j) for(int k=0;k<n;++k)
        {
            double s=0.0;
            for(int i=0;i<m;++i) 
            {
              s+=A[i+j*m]*A[i+k*m];
            }
            AtA[j+k*n]=s;
        }
        // RtR
        for(int j=0;j<n;++j) for(int k=0;k<n;++k)
        {
            double s=0.0;
            for(int i=0;i<n;++i) 
            {
              s+=R[i+j*n]*R[i+k*n];
            }
            RtR[j+k*n]=s;
        }

        // rel error
        double num=0.0, den=0.0;
        for(int i=0;i<n*n;++i)
        { 
          double d=AtA[i]-RtR[i]; 
          num+=d*d; 
          den+=AtA[i]*AtA[i]; 
        }
        double rel=sqrt(num)/(sqrt(den)+1e-32);
        // printf("TSQR check: ||A^T A - R^T R||_F / ||A^T A||_F = %.3e\n", rel);
        free(RtR);
        free(AtA);
    }

    free(A_local);
    if (rank==0) 
  {
        free(R);
        free(sendbuf);
        free(A);
    }

    MPI_Finalize();
    return 0;
}