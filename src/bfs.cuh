// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include "fused_spmv.cuh"

//#define NBLOCKS 16384
#define NTHREADS 1024

using namespace mgpu;

void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult, CudaContext& context) {
    /*SpmvKernel<float>(d_csrValA,
                      d_csrColIndA,
                      d_csrRowPtrA,
                      d_inputVector,
                      d_spmvResult,
                      m,
                      edge,
                      context);*/
    /*const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

    //cusparseStatus_t status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, alpha, descr, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_inputVector, beta, d_spmvResult);

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

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);*/
}

__global__ void addResult( int *d_bfsResult, float *d_spmvResult, float *d_next, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
        d_next[idx] = 0;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

void bfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth, CudaContext& context) {

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    int *h_bfsResult;
    float *h_spmvResult;
    h_bfsResult = (int*)malloc(m*sizeof(int));
    h_spmvResult = (float*)malloc(m*sizeof(float));

    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=-1;
        h_spmvResult[i]=0;
        if( i==vertex ) {
            h_bfsResult[i]=0;
            h_spmvResult[i]=1;
        }
    }
    //std::cout << "This is m: " << m << std::endl;
    //print_array(h_bfsResult,m);
    cudaMemcpy(d_spmvSwap,h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfsResult,h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate values for BFS (csrValA where everything is 1)
    float *h_bfsValA, *d_bfsValA;
    h_bfsValA = (float*)malloc(edge*sizeof(float));
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        h_bfsValA[i] = 1;
    }
    cudaMemcpy(d_bfsValA, h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);
    
    // Use Thrust to generate initial vector
    //thrust::device_ptr<float> dev_ptr(d_inputVector);
    //thrust::fill(dev_ptr, dev_ptr + m, (float) 0);
    //thrust::device_vector<float> inputVector(m, 0);
    //inputVector[vertex] = 1;
    //d_inputVector = thrust::raw_pointer_cast( &inputVector[0] );

    //Preprocessing, compute neighbor_len, set interval_value, and use interval_expand
    //to set d_node_id
    int *d_neighbor_len;
    int *d_interval_value;
    int *d_node_id;
    cudaMalloc((void**)&d_neighbor_len, m*sizeof(int));
    cudaMalloc((void**)&d_interval_value, m*sizeof(int));
    cudaMalloc((void**)&d_node_id, edge*sizeof(int));
    int thread_num = 1024;
    int block_num = (edge+thread_num-1)/thread_num;
    GetNeighborListLen<<<block_num, thread_num>>>(d_csrRowPtrA, d_neighbor_len, d_interval_value, m);
    Scan<mgpu::MgpuScanTypeExc>((int*)d_neighbor_len, m, (int)0, mgpu::plus<int>(), (int*)0, (int*)0, (int*)d_neighbor_len, context);
    IntervalExpand(edge, d_neighbor_len, d_interval_value, m, d_node_id, context);

    int *h_node_id = (int*)malloc(edge*sizeof(int));
    float *h_result = (float*)malloc(m*sizeof(float));
    cudaMemcpy(h_node_id, d_node_id, edge*sizeof(int), cudaMemcpyDeviceToHost); 

    free(h_node_id);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    /*spmv(d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
    
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1, m);

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<3; i++ ) {
        if( i%2==0 ) {
            spmv( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i, m);
        } else {
            spmv( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i, m);
        }
    }*/
    for (int i = 1; i < depth; ++i) {
        if (i%2==0) {
            ComputeResult<<<block_num, thread_num>>>(d_bfsValA, d_csrColIndA, d_spmvResult, d_node_id, d_spmvSwap, edge);
            cudaMemcpy(h_result, d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            addResult<<<(m+thread_num-1)/thread_num, thread_num>>>(d_bfsResult, d_spmvSwap, d_spmvResult, i, m);
        } else {
            ComputeResult<<<block_num, thread_num>>>(d_bfsValA, d_csrColIndA, d_spmvSwap, d_node_id, d_spmvResult, edge);
            cudaMemcpy(h_result, d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            addResult<<<(m+thread_num-1)/thread_num, thread_num>>>(d_bfsResult, d_spmvResult, d_spmvSwap, i, m);
        }
        /*printf("%d: ", i);
        for (int i = 0; i < m; ++i) {
            printf("%f ", h_result[i]);
        }
        printf("\n");*/
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("GPU BFS finished in %f msec. \n", elapsed);

    //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_spmvResult,m);
    //print_array(h_bfsResult,m);
    //
    
    free(h_result);
    cudaFree(d_neighbor_len);
    cudaFree(d_interval_value);
    cudaFree(d_node_id);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}
