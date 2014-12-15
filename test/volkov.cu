// Written by Vasily Volkov.
// Copyright (c) 2008, The Regents of the University of California. 
// All rights reserved.

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>

#define TIMER_TOLERANCE 0.1f

#define BEGIN_TIMING( )	\
{\
    unsigned int n_iterations;	\
    for( n_iterations = 1; n_iterations < 0x80000000; n_iterations *= 2 )\
    {\
        Q( cudaThreadSynchronize( ) );\
        Q( cudaEventRecord( start, 0 ) );\
        for( unsigned int iteration = 0; iteration < n_iterations; iteration++ ){

#define END_TIMING( seconds ) }\
        Q( cudaEventRecord( end, 0 ) );\
        Q( cudaEventSynchronize( end ) );\
        float milliseconds;\
        Q( cudaEventElapsedTime( &milliseconds, start, end ) );\
        seconds = milliseconds/1e3f;\
        if( seconds >= TIMER_TOLERANCE )\
            break;\
    }\
    seconds /= n_iterations;\
}

#define Q( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

__device__ void saxpy( float a, float *b, float *c )
{
    c[0] += a*b[0];
    c[1] += a*b[1];
    c[2] += a*b[2];
    c[3] += a*b[3];
    c[4] += a*b[4];
    c[5] += a*b[5];
    c[6] += a*b[6];
    c[7] += a*b[7];
    c[8] += a*b[8];
    c[9] += a*b[9];
    c[10] += a*b[10];
    c[11] += a*b[11];
    c[12] += a*b[12];
    c[13] += a*b[13];
    c[14] += a*b[14];
    c[15] += a*b[15];
}

__device__ void saxpyminplus( float a, float *b, float *c )
{
    c[0] = min(c[0],a+b[0]);
    c[1] = min(c[1],a+b[1]);
    c[2] = min(c[2],a+b[2]);
    c[3] = min(c[3],a+b[3]);
    c[4] = min(c[4],a+b[4]);
    c[5] = min(c[5],a+b[5]);
    c[6] = min(c[6],a+b[6]);
    c[7] = min(c[7],a+b[7]);
    c[8] = min(c[8],a+b[8]);
    c[9] = min(c[9],a+b[9]);
    c[10] = min(c[10],a+b[10]);
    c[11] = min(c[11],a+b[11]);
    c[12] = min(c[12],a+b[12]);
    c[13] = min(c[13],a+b[13]);
    c[14] = min(c[14],a+b[14]);
    c[15] = min(c[15],a+b[15]);
}

extern "C" __global__ void sgemmNT( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * 64;
    const int iby = blockIdx.y * 16;
    const int id = inx + iny*16;

    A += ibx + id;
    B += iby + inx + __mul24( iny, ldb );
    C += ibx + id  + __mul24( iby, ldc );
    const float *Blast = B + k*ldb;

    float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __shared__ float bs[16][16];
    do
    {
#pragma unroll
        for( int i = 0; i < 16; i += 4 )
            bs[iny+i][inx]  = B[i*ldb];
        __syncthreads();

#pragma unroll
        for( int i = 0; i < 16; i++, A += lda )
            saxpy( A[0], &bs[i][0], c ); 

        B += 16*ldb;
        __syncthreads();
    } while( B < Blast );

    for( int i = 0; i < 16; i++, C += ldc )
        C[0] = alpha*c[i] + beta*C[0]; 
}	

extern "C" __global__ void sgemmNN( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * 64;
    const int iby = blockIdx.y * 16;
    const int id = inx + iny*16;

    A += ibx + id;
    B += inx + __mul24( iby + iny, ldb );
    C += ibx + id  + __mul24( iby, ldc );

    const float *Blast = B + k;

    float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __shared__ float bs[16][17];
    do
    {
#pragma unroll
        for( int i = 0; i < 16; i += 4 )
            bs[inx][iny+i]  = B[i*ldb];
        __syncthreads();

#pragma unroll
        for( int i = 0; i < 16; i++, A += lda )
            saxpy( A[0], &bs[i][0], c ); 

        B += 16;
        __syncthreads();
    } while( B < Blast );

    for( int i = 0; i < 16; i++, C += ldc )
        C[0] = alpha*c[i] + beta*C[0]; 
}	

extern "C" void ourSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{	
    dim3 grid( m/64, n/16 ), threads( 16, 4 );
    if( transb == 'N' || transb == 'n' )
        sgemmNN<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
    else
        sgemmNT<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
}	

//
//	auxiliary routines
//	
void fill( float *A, int n, int maxi )
{	
    for( int j = 0; j < n; j++ )
        A[j] = float( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );
}	

float diff( int m, int n, float *A, int lda, float *B, int ldb )
{
    float err = 0;
    for( int j = 0; j < n; j++ )
        for( int i = 0; i < m; i++ )
            err = max( err, fabs( A[i+j*lda] - B[i+j*ldb] ) );
    return err;
}

//
//	main()
//
int main( int argc, char **argv )
{	
    const int N = 4096;

    //
    //  startup
    //
    int idevice = 0;
    for( int i = 1; i < argc-1; i ++ )
        if( strcmp( argv[i], "-device" ) == 0 )
            idevice = atoi( argv[i+1] );

    Q( cudaSetDevice( idevice ) );

    struct cudaDeviceProp prop;
    Q( cudaGetDeviceProperties( &prop, idevice ) );
    printf( "\nDevice: %s, %.0f MHz clock, %.0f MB memory.\n", prop.name, prop.clockRate/1000.f, prop.totalGlobalMem/1024.f/1024.f );

    cudaEvent_t start, end;
    Q( cudaEventCreate( &start ) );
    Q( cudaEventCreate( &end ) );

    Q( cublasInit( ) );

    //
    //  allocate memory
    //
    float *A = (float*)malloc( N*N*sizeof( float ) );
    float *B = (float*)malloc( N*N*sizeof( float ) );
    float *C = (float*)malloc( N*N*sizeof( float ) );
    float *cublas_result = (float*)malloc( N*N*sizeof( float ) );
    float *our_result = (float*)malloc( N*N*sizeof( float ) );

    fill( A, N*N, 31 );
    fill( B, N*N, 31 );
    fill( C, N*N, 31 );

    float *dA, *dB, *dC;
    Q( cublasAlloc( N*N, sizeof(float), (void**)&dA ) );
    Q( cublasAlloc( N*N, sizeof(float), (void**)&dB ) );
    Q( cublasAlloc( N*N, sizeof(float), (void**)&dC ) );
    Q( cudaMemcpy( dA, A, N*N*sizeof(float), cudaMemcpyHostToDevice ) );
    Q( cudaMemcpy( dB, B, N*N*sizeof(float), cudaMemcpyHostToDevice ) );
    	
    //
    //	bench square matrices
    //
    for( int i = 0; i < 2; i++ )
    {
        const char transa = 'N';
        const char transb = i ? 'T' : 'N';

        printf( "\ntesting sgemm( '%c', '%c', n, n, n, ... )\n\n", transa, transb );

        const int nb = 64;
        printf( "   n   CUBLAS,Gflop/s   we,Gflop/s   \"error\"\n" );
        for( int idim = 1; idim <= N/nb; idim = int((idim+1)*1.1) )
        {
            int dim = idim*nb;

            //
            //	set up the parameters
            //
            const int m = dim;
            const int n = dim;
            const int k = dim;
            const int lda = dim;
            const int ldb = dim;
            const int ldc = dim;
            const float alpha = 1;
            const float beta = -1;

            //
            // compute with CUBLAS
            //
            Q( cublasSetMatrix( m, n, sizeof( float ), C, ldc, dC, ldc ) );
            cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            Q( cublasGetError( ) );
            Q( cublasGetMatrix( m, n, sizeof( float ), dC, ldc, cublas_result, ldc ) );

            //
            // compute with our routine
            //
            Q( cublasSetMatrix( m, n, sizeof( float ), C, ldc, dC, ldc ) );
            ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            Q( cublasGetMatrix( m, n, sizeof( float ), dC, ldc, our_result, ldc ) );

            //
            //	check the difference in results
            //
            float difference = diff( m, n, cublas_result, ldc, our_result, ldc );

            //
            //	bench cublas
            //
            double cublas_time;
            cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            BEGIN_TIMING( );
            cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            END_TIMING( cublas_time );

            double cublas_gflops = 2.*m*n*k/cublas_time/1e9;

            //
            //	bench our routine
            //
            double our_time;
            ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            BEGIN_TIMING( );
            ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
            END_TIMING( our_time );

            double our_gflops = 2.*m*n*k/our_time/1e9;

            //
            //	report the results
            //
            printf( "%5d %11.2f %14.2f %8g\n", n, cublas_gflops, our_gflops, difference );
        }
    }
	
    //
    //	shutdown
    //

    cublasFree( dA );
    cublasFree( dB );
    cublasFree( dC );

    free( A );
    free( B );
    free( C );

    free( cublas_result );
    free( our_result );

    Q( cublasShutdown( ) );

    return 0;
}	
