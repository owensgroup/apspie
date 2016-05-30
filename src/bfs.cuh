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

__device__ void swap(int &first, int &second) {
    int temp;
    temp = first;
    first = second;
    second = temp;
    return;
}

__global__ void generateHistogram( const int new_n, const int nnz, const int *d_spmvSwapInd, int *d_sendHist, int *d_sendHistProc, int *d_counter, int *d_mutex ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<nnz-1; idx+=blockDim.x*gridDim.x ) {
        if( d_spmvSwapInd[idx]/new_n > d_spmvSwapInd[idx+1]/new_n ) {
            //printf("<=\n");
            bool isSet = false;
            do {
                if( isSet = atomicCAS( d_mutex, 0, 1 ) == 0 ) {
                    d_sendHist[*d_counter] = idx;
                    d_sendHistProc[*d_counter] = d_spmvSwapInd[idx]/new_n;
                    if( *d_counter>0 ) {
                        int i;
    					for( i = *d_counter; i > 0 && d_sendHistProc[i] < d_sendHistProc[i-1]; i--) {
                            swap( d_sendHistProc[i], d_sendHistProc[i-1] );
                        }
                        swap( d_sendHist[*d_counter], d_sendHistProc[i] );
                    } 
                    atomicAdd(d_counter, 1);
                } if( isSet ) {
                    *d_mutex = 0;
                }
            } while( !isSet );
        }
    }
}

template< typename T >
void bfsSparse( const int vertex, const int new_nnz, const int new_n, const int old_n, const int multi, const int rank, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, new_nnz, old_n );

    // Allocate GPU memory for result
    int *h_bfsResult = (int*)malloc(old_n*sizeof(int));

    int *h_spmvResultInd = (int*)malloc(old_n*sizeof(int));
    int *h_spmvSwapInd = (int*)malloc(old_n*sizeof(int));
    float *h_spmvResultVec = (float*)malloc(old_n*sizeof(float));
    float *h_spmvSwapVec = (float*)malloc(old_n*sizeof(float));

    int *d_spmvResultInd, *d_spmvSwapInd;
    float *d_spmvResultVec, *d_spmvSwapVec;
    cudaMalloc(&d_spmvResultInd, old_n*sizeof(int));
    cudaMalloc(&d_spmvSwapInd, old_n*sizeof(int));
    cudaMalloc(&d_spmvResultVec, old_n*sizeof(float));
    cudaMalloc(&d_spmvSwapVec, old_n*sizeof(float));
    
    int h_nnz = 1;
    int h_cscVecCount = 0;

    // Allocate histogram of send 
    int *h_sendHist = (int*)malloc(multi*sizeof(int));
    int *h_sendHistProc = (int*)malloc(multi*sizeof(int));
    int *h_recvHist = (int*)malloc(multi*sizeof(int));

    int *d_sendHist, *d_sendHistProc, *d_recvHist;
    cudaMalloc(&d_sendHist, multi*sizeof(int));
    cudaMalloc(&d_sendHistProc, multi*sizeof(int));
    cudaMalloc(&d_recvHist, multi*sizeof(int));

    // Only make one element of h_bfsResult 0 if following is true:
    //   a) if not last processor
    //   b) if last processor
    // Also generate initial sparse vector using vertex
    for( int i=0; i<new_n; i++ ) {
        h_bfsResult[i] = -1;
        if( ((rank != multi-1 && rank == vertex/new_n) || (rank==multi-1 && vertex >= rank*(old_n/multi+1)) ) && i==vertex ) {
            h_bfsResult[vertex-rank*(old_n/multi+1)] = 0;
            h_spmvResultInd[0] = vertex-rank*(old_n/multi+1)
			h_spmvResultVec[0] = 1.0;
            printf("Source vertex on processor %d!\n", rank);
        }
    }

    // Generate d_ones, d_index
    for( int i=0; i<old_n; i++ ) {
        d->h_ones[i] = 1;
        d->h_index[i] = i;
    }
	//print_array(d->h_spmvResult, 40);
    cudaMemcpy(d_bfsResult, h_bfsResult, new_n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultInd, h_spmvResultInd, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultVec, h_spmvResultVec, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, new_n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, new_n*sizeof(int), cudaMemcpyHostToDevice);

    // Allocate counter and mutex (for histogram)
    int *h_counter = (int*)malloc(sizeof(int));
    *h_counter = 0;
    int *d_counter;
    int *d_mutex;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_mutex, sizeof(int));
    cudaMemcpy(d_mutex, h_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (new_n+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, new_n);

	// Testing correctness:
	printf("Rank %d, new_nnz %d, new_n %d, old_n %d, multi %d\n", rank, new_nnz, new_n, old_n, multi);

    // Keep a count of new_nnzs traversed
    int cumsum = 0;
    int sum = 0;
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    for( int i=1; i<2; i++ ) {
    //for( int i=1; i<depth; i++ ) {
        //printf("Iteration %d\n", i);
            //spmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //cuspmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, handle, descr);

            // op=1 Arithmetic semiring
            sum = mXvSparse( d_spmvResultInd, d_spmvResultVec, new_nnz, new_n, old_n, h_nnz, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwapInd, d_spmvSwapVec, d, context);

            // Generate the send histograms
            /*cudaMemcpy(d_counter, h_counter, sizeof(int), cudaMemcpyHostToDevice);
            generateHistogram<<<NTHREADS, NBLOCKS>>>( old_n/multi+1, h_nnz, d_spmvSwapInd, d_sendHist, d_sendHistProc, d_counter, d_mutex );

            cudaMemcpy( h_sendHist, d_sendHist, multi*sizeof(int), cudaMemcpyDeviceToHost);
            print_array(h_sendHist, multi);

            // Exchange then sum up histograms
            MPI_Alltoall( h_sendHist, 1, MPI_INT, h_recvHist, 1, MPI_INT, MPI_COMM_WORLD );
            int sumHist = 0;
            for( int i=0; i<multi; i++ ) sumHist += h_recvHist[i];

            // Exchange vectors
            MPI_Alltoallv( d_spmvSwapInd, h_sendHist, h_scanHist, MPI_INT, d_spmvRecvInd, h_recvHist, MPI_INT, MPI_COMM_WORLD );
            MPI_Alltoallv( d_spmvSwapVec, h_sendHist, h_scanHist, MPI_INT, d_spmvRecvVec, h_recvHist, MPI_INT, MPI_COMM_WORLD );

            // Merge vectors
            MergesortPairs(d_spmvRecv)

            // Update BFS Result
            addResultSparse<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwapInd, i, h_nnz);

            // Prune new vector
            bitifySparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapInd, d->d_randVecInd, h_nnz );
            mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_randVecInd, h_nnz, 0, mgpu::plus<int>(), (int*)0, &h_cscVecCount, d->d_cscColGood, context );
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapInd, d->d_randVecInd, d->d_cscColGood, d_spmvResultInd, h_nnz );
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapVec, d->d_randVecInd, d->d_cscColGood, d_spmvResultVec, h_nnz );           
            h_nnz = h_cscVecCount;

            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResultInd,d_spmvResultInd, h_nnz*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResultInd,h_nnz);
        cumsum+=sum;*/
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
