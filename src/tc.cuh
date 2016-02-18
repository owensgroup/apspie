// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "mXv.cuh"
#include "scratch.hpp"

#define NTHREADS 512

// Uses MGPU SpMV
template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

// Uses cuSPARSE SpMV
template<typename T>
void cuspmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, cusparseHandle_t handle, cusparseMatDescr_t descr ) {

    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsrmv(handle,                   
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              m, m, edge, 
                              alpha, descr, 
                              d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                              d_inputVector, beta, d_spmvResult );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("spmv multiplication successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("Error: Matrix type not supported.\n");
    }

}

// Uses cuSPARSE SpGEMM
template<typename T>
int spgemm( const int edge, const int m, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, const T* d_cscValB, const int *d_cscColPtrB, const int *d_cscRowIndB, T* &d_cscValC, int* &d_cscColPtrC, int* &d_cscRowIndC ) {
    
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    cudaMalloc((void**) &d_cscColPtrC, (m+1)*sizeof(int));
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseStatus_t status = cusparseXcsrgemmNnz( handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              m, m, m,
                              descr, edge,
                              d_cscColPtrA, d_cscRowIndA,
                              descr, edge,
                              d_cscColPtrB, d_cscRowIndB,
                              descr,
                              d_cscColPtrC, nnzTotalDevHostPtr );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("nnz count successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("Error: Matrix type not supported.\n");
    }
    if( NULL != nnzTotalDevHostPtr )
        nnzC = *nnzTotalDevHostPtr;
    else {
        cudaMemcpy( &nnzC, d_cscColPtrC+m, sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( &baseC, d_cscColPtrC, sizeof(int), cudaMemcpyDeviceToHost );
        nnzC -= baseC;
    }
    //printf("Matrix C: %d nnz\n", nnzC);
    cudaMalloc((void**) &d_cscRowIndC, nnzC*sizeof(int));
    cudaMalloc((void**) &d_cscValC, nnzC*sizeof(T));

    status                  = cusparseScsrgemm( handle,                   
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              m, m, m,
                              descr, edge,
                              d_cscValA, d_cscColPtrA, d_cscRowIndA, 
                              descr, edge,
                              d_cscValB, d_cscColPtrB, d_cscRowIndB,
                              descr,
                              d_cscValC, d_cscColPtrC, d_cscRowIndC );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("spgemm multiplication successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("Error: Matrix type not supported.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    return nnzC;
}

// TODO: move TC code to here
template< typename T >
__global__ void ewiseMultTc( const int edge, const int m, const T *d_cscValD, const int *d_cscColPtrD, const int *d_cscRowIndD, const T *d_cscValC, const int *d_cscColPtrC, const int *d_cscRowIndC, T *d_cscVecVal ) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    T count = 0;
    int i, k;
    int i_end, k_end;
    int i_row, k_row;
    for (int idx = tid; idx < m; idx += stride) {
        i = d_cscColPtrC[idx];
        k = d_cscColPtrD[idx];
        i_end = d_cscColPtrC[idx+1];
        k_end = d_cscColPtrD[idx+1];

        i_row = d_cscRowIndC[i];
        k_row = d_cscRowIndD[k];
        while( i<i_end && k<k_end && k_row < idx ) {
            int diff = i_row - k_row;
            if( diff == 0 )
                count += d_cscValD[k];
            if( diff <= 0 ) {
                i++;
                i_row = d_cscRowIndC[i];
            }
            if( diff >= 0 ) {
                k++;
                k_row = d_cscRowIndD[k];
            }
        }
    }
    d_cscVecVal[tid] = count;
}

__global__ void addResult( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        //d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
        if( d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0 ) {
            d_bfsResult[idx] = iter;
        } else d_spmvResult[idx] = 0;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

template< typename T >
void bfs( const int vertex, const int edge, const int m, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, edge, m );

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        d->h_bfsResult[i]=-1;
        d->h_spmvResult[i]=0.0;
        d->h_ones[i] = 1;
        d->h_index[i] = i;
        if( i==vertex ) {
            d->h_bfsResult[i]=0;
            d->h_spmvResult[i]=1.0;
        }
    }
    cudaMemcpy(d_bfsResult, d->h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvSwap, d->h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, m);

    // Generate values for BFS (cscValA where everything is 1)
    float *d_bfsValA;
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        d->h_bfsValA[i] = 1.0;
    }
    cudaMemcpy(d_bfsValA, d->h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    //spmv<float>(d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, context);
    //cuspmv<float>(d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, handle, descr);
    mXv<float>(d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, d, context);

    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1, m);

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<5; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //cuspmv<float>( d_spmvResult, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, handle, descr);
            mXv<float>( d_spmvResult, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, context);
            //cuspmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, handle, descr);
            mXv<float>( d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        }
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);
    //printf("The maximum frontier size was: %d.\n", frontier_max);
    //printf("The average frontier size was: %d.\n", frontier_sum/depth);

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
}

template< typename T >
int mXm( const int edge, const int m, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, const T* d_cscValB, const int *h_cscColPtrB, const int *d_cscColPtrB, const int *d_cscRowIndB, T *d_cscValC, int *d_cscColPtrC, int *d_cscRowIndC, mgpu::CudaContext& context) {

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, edge, m );

    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        d->h_ones[i] = 1;
        d->h_index[i] = i;
    }
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_cscColPtrC, (m+1)*sizeof(int));
    cudaMalloc((void**) &d_cscRowIndC, 16*edge*sizeof(int));
    cudaMalloc((void**) &d_cscValC, 16*edge*sizeof(T));

    // Generate d_cscColDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, m);

    // Initialize nnz cumulative
    int *h_cscColPtrC = (int*)malloc((m+1)*sizeof(int));
    int total_nnz = 0;
    h_cscColPtrC[0] = total_nnz;
    int nnz = 0;

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    for( int i=0; i<m; i++ ) {
    //for( int i=0; i<2; i++ ) {
        nnz = h_cscColPtrB[i+1]-h_cscColPtrB[i];
        //printf("Reading %d elements in matrix B: %d to %d\n", nnz, h_cscColPtrB[i], h_cscColPtrB[i+1]);
        if( nnz ) {
        mXv<T>(&d_cscRowIndB[h_cscColPtrB[i]], &d_cscValB[h_cscColPtrB[i]], edge, m, nnz, d_cscValA, d_cscColPtrA, d_cscRowIndA, &d_cscRowIndC[total_nnz], &d_cscValC[total_nnz], d, context);
        total_nnz += nnz;
        h_cscColPtrC[i+1] = total_nnz;
        //printf("mXv iteration %d: ColPtrC at %d\n", i, total_nnz);
        /*cudaMemcpy(d->h_bfsResult, d_cscRowIndC, total_nnz*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_bfsResult,total_nnz);
        cudaMemcpy(d->h_spmvResult, d_cscValC, total_nnz*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_spmvResult,total_nnz);*/
        }
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU mXm finished in %f msec. \n", elapsed);

    int *h_cscRowIndC = (int*)malloc(total_nnz*sizeof(int));
    T *h_cscValC = (T*)malloc(total_nnz*sizeof(T));
    cudaMemcpy(h_cscRowIndC, d_cscRowIndC, total_nnz*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValC, d_cscValC, total_nnz*sizeof(T), cudaMemcpyDeviceToHost);
    print_matrix(h_cscValC, h_cscColPtrC, h_cscRowIndC, m);

    return total_nnz;
}
