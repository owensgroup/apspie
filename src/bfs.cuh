// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>

//#define NBLOCKS 16384
#define NTHREADS 1024

void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult, cusparseHandle_t handle, cusparseMatDescr_t descr ) {
    const float alf = 1;
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
    }*/
}

__global__ void addResult( int *d_bfsResult, const float *d_spmvResult, const int iter ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

void nnz( const int m, const float *A, int *nnzPerRowColumn, int *nnzTotalDevHostPtr, cusparseHandle_t handle, cusparseMatDescr_t descr ) {

    cusparseStatus_t status = cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, 1, m, descr, A, 1, nnzPerRowColumn, nnzTotalDevHostPtr);

    /*switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("nnz count successful!\n");
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
    }*/
}

void bfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    int *d_nnzPerRowColumn;
    int *d_nnzTotalDevHostPtr;

    cudaMalloc(&d_nnzPerRowColumn, sizeof(int));
    cudaMalloc(&d_nnzTotalDevHostPtr, sizeof(int));

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

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();
    spmv(d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, handle, descr);
    int *frontier;
    int frontier_max;
    int frontier_sum;
    frontier = (int*)malloc(sizeof(int));

    nnz(m, d_spmvResult, d_nnzPerRowColumn, d_nnzTotalDevHostPtr, handle, descr);
    cudaMemcpy(frontier,d_nnzPerRowColumn,sizeof(int),cudaMemcpyDeviceToHost);
    frontier_max = *frontier;
    frontier_sum = *frontier;
    printf("[1]:%d ", *frontier);
    
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1 );

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<3; i++ ) {
        if( i%2==0 ) {
            spmv( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, handle, descr );
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i );
            if( i<10 ) {
                nnz(m, d_spmvSwap, d_nnzPerRowColumn, d_nnzTotalDevHostPtr, handle, descr);
                cudaMemcpy(frontier,d_nnzPerRowColumn,sizeof(int),cudaMemcpyDeviceToHost);
                frontier_max = (*frontier > frontier_max) ? *frontier : frontier_max;
                frontier_sum += *frontier;
                printf("[%d]:%d ", i, *frontier);
            }
        } else {
            spmv( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, handle, descr );
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i );
            if( i<10 ) {
                nnz(m, d_spmvResult, d_nnzPerRowColumn, d_nnzTotalDevHostPtr, handle, descr);
                cudaMemcpy(frontier,d_nnzPerRowColumn,sizeof(int),cudaMemcpyDeviceToHost);
                frontier_max = (*frontier > frontier_max) ? *frontier : frontier_max;
                frontier_sum += *frontier;
                printf("[%d]:%d ", i, *frontier);
             }
        }
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);
    printf("The maximum frontier size was: %d.\n", frontier_max);
    printf("The average frontier size was: %d.\n", frontier_sum/depth);

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}
