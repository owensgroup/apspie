// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include "spmv.cuh"

//#define NBLOCKS 16384
#define NTHREADS 1024

using namespace mgpu;

void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult, CudaContext& context) {
    SpmvKernel<float>(d_csrValA,
                      d_csrColIndA,
                      d_csrRowPtrA,
                      d_inputVector,
                      d_spmvResult,
                      m,
                      edge,
                      context);
    /*const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

    //cusparseStatus_t status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, alpha, descr, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_inputVector, beta, d_spmvResult);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsrmv(handle,                   
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              m, m, edge, 
                              alpha, descr, 
                              d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                              d_inputVector, beta, d_spmvResult );

    /*switch( status ) {
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

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);*/
}

__global__ void addResult( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        //d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
        if( d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

void bfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth[], CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    int *h_bfsResult;
    float *h_spmvResult;
    h_bfsResult = (int*)malloc(m*sizeof(int));
    h_spmvResult = (float*)malloc(m*sizeof(float));

    //std::cout << "This is m: " << m << std::endl;
    //print_array(h_bfsResult,m);

    // Generate values for BFS (csrValA where everything is 1)
    float *h_bfsValA, *d_bfsValA;
    h_bfsValA = (float*)malloc(edge*sizeof(float));
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    // Use Thrust to generate initial vector
    //thrust::device_ptr<float> dev_ptr(d_inputVector);
    //thrust::fill(dev_ptr, dev_ptr + m, (float) 0);
    //thrust::device_vector<float> inputVector(m, 0);
    //inputVector[vertex] = 1;
    //d_inputVector = thrust::raw_pointer_cast( &inputVector[0] );

    for( int i=0; i<edge; i++ ) {
        h_bfsValA[i] = 1;
    }

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    //cudaProfilerStart();

    for( int test=0;test<10;test++) {
    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=-1;
        h_spmvResult[i]=0;
        if( i==test ) {
            h_bfsResult[i]=0;
            h_spmvResult[i]=1;
        }
    }

    cudaMemcpy(d_spmvSwap,h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfsResult,h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfsValA, h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);
    
    gpu_timer.Start();
    spmv(d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
    
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1, m);

    for( int iter=2; iter<depth[test]; iter++ ) {
        if( iter%2==0 ) {
            spmv( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, iter, m);
        } else {
            spmv( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, iter, m);
        }
    }
   
    //cudaProfilerStop();
    gpu_timer.Stop();
    elapsed = gpu_timer.ElapsedMillis();
    //printf("GPU BFS finished in %f msec. \n", elapsed);
    //printf("The maximum frontier size was: %d.\n", frontier_max);
    //printf("The average frontier size was: %d.\n", frontier_sum/depth);
    printf("%f\n", elapsed);
    }
    printf("bfs done\n");

    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}
