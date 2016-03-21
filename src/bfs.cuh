// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "mXv.cuh"
#include "scratch.hpp"

//#define NBLOCKS 16384
#define NTHREADS 512

template<typename T>
void spmv( const T *d_inputVector, const int new_nnz, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, new_nnz, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

__global__ void bitifySparse( const int *d_randVec, int *d_randVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_randVec[idx]>-1 ) d_randVecInd[idx] = 1;
        else d_randVecInd[idx] = 0;
    }
}

__global__ void addResultSparse( int *d_bfsResult, int* d_spmvSwapInd, const int iter, const int length) {
    const int STRIDE = gridDim.x * blockDim.x;
    for( int idx = (blockIdx.x * blockDim.x ) + threadIdx.x; idx<length; idx+=STRIDE ) {
        if( d_spmvSwapInd[idx]>=0 && d_bfsResult[d_spmvSwapInd[idx]] < 0 ) {
            d_bfsResult[d_spmvSwapInd[idx]] = iter;
        } else d_spmvSwapInd[idx] = -1;
    }
}

template< typename T >
__global__ void streamCompactSparse( const T *d_tempVal, const int *d_cscFlag, const int *d_cscColGood, T *d_cscVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_cscFlag[idx] ) d_cscVecInd[d_cscColGood[idx]]=d_tempVal[idx];
}

template< typename T >
void bfsSparse( const int vertex, const int new_nnz, const int new_m, const int old_m, const int multi, const int rank, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, new_nnz, old_m );

    // Allocate GPU memory for result
    int *h_bfsResult = (int*)malloc(m*sizeof(int));

    int *h_spmvResultInd = (int*)malloc(old_m*sizeof(int));
    int *h_spmvSwapInd = (int*)malloc(old_m*sizeof(int));
    float *h_spmvResultVec = (float*)malloc(old_m*sizeof(float));
    float *h_spmvSwapVec = (float*)malloc(old_m*sizeof(float));

    int *d_spmvResultInd, *d_spmvSwapInd;
    float *d_spmvResultVec, *d_spmvSwapVec;
    cudaMalloc(&d_spmvResultInd, old_m*sizeof(int));
    cudaMalloc(&d_spmvSwapInd, old_m*sizeof(int));
    cudaMalloc(&d_spmvResultVec, old_m*sizeof(float));
    cudaMalloc(&d_spmvSwapVec, old_m*sizeof(float));
    
    int h_nnz = 1;
    int h_cscVecCount = 0;

    // Allocate histogram of send 
    int *h_sendHist = (int)malloc(multi*sizeof(int));
    int *h_recvHist = (int)malloc(multi*sizeof(int));

    // Generate initial vector using vertex
    h_spmvResultInd[0] = vertex;
    h_spmvResultVec[0] = 1.0;
    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=-1;
        d->h_ones[i] = 1;
        d->h_index[i] = i;
        if( i==vertex )
            h_bfsResult[i] = 0;
    }
	//print_array(d->h_spmvResult, 40);
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultInd, h_spmvResultInd, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultVec, h_spmvResultVec, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, m);

    // Keep a count of new_nnzs traversed
    int cumsum = 0;
    int sum = 0;
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    for( int i=1; i<depth; i++ ) {
        //printf("Iteration %d\n", i);
            //spmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //cuspmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, handle, descr);

            // op=1 Arithmetic semiring
            sum = mXvSparse( d_spmvResultInd, d_spmvResultVec, new_nnz, m, nnz, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwapInd, d_spmvSwapVec, d, context);

            // Generate the send histograms
            generateHistogram<<<1, multi>>>( d_hist, d_spmvSwapInd, h_sendHist, h_recvHist );

            // Exchange then sum up histograms
            MPI_Alltoallv( h_sendHist, 1, d->h_index, MPI_INT, h_recvHist, d->h_index, MPI_INT, MPI_COMM_WORLD );
            int sumHist = 0;
            for( int i=0; i<multi; i++ ) sumHist += h_recvHist[i];

            // Exchange vectors
            MPI_Alltoallv( d_spmvSwapInd, h_sendHist, h_scanHist, MPI_INT, d_spmvRecvInd, h_recvHist, MPI_INT, MPI_COMM_WORLD );

            // Merge vectors
            MergesortPairs(d_spmvRecv

            // Update BFS Result
            addResultSparse<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwapInd, i, nnz);

            // Prune new vector
            bitifySparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapInd, d->d_randVecInd, nnz );
            mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_randVecInd, nnz, 0, mgpu::plus<int>(), (int*)0, &h_cscVecCount, d->d_cscColGood, context );
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapInd, d->d_randVecInd, d->d_cscColGood, d_spmvResultInd, nnz );
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapVec, d->d_randVecInd, d->d_cscColGood, d_spmvResultVec, nnz );           
            nnz = h_cscVecCount;

            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResultInd,d_spmvResultInd, nnz*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResultInd,nnz);
        cumsum+=sum;
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);
    printf("Traversed new_nnzs: %d\n", cumsum);
    printf("Performance: %f GTEPS\n", (float)cumsum/(elapsed*1000000));

    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);
}
