// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
//#include "spmv.cuh"

//#define NBLOCKS 16384
#define NTHREADS 1024

/*void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult, CudaContext& context) {
    SpmvKernel<float>(d_csrValA,
                      d_csrColIndA,
                      d_csrRowPtrA,
                      d_inputVector,
                      d_spmvResult,
                      m,
                      edge,
                      context);
}*/

__global__ void updateBfs( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        if( d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void diff( const int *d_csrRowPtr, int *d_csrRowDiff, const int m) {

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<m; idx+=blockDim.x*gridDim.x) {
        d_csrRowDiff[idx] = d_csrRowPtr[idx+1]-d_csrRowPtr[idx];
    }
}
 
void gather( const int h_csrVecCount, const int * d_csrVecInd, const int *d_csrRowPtr, int *d_csrRowGood, CudaContext& context ) {
    MGPU_MEM(int) insertdata = context.FillAscending( h_csrVecCount, 0, 1 );
    IntervalGather( h_csrVecCount, d_csrVecInd, insertdata->get(), h_csrVecCount, d_csrRowPtr, d_csrRowGood, context);
}

__global__ void lookRight( const int *d_csrSwapInd, const int total, float *d_csrFlagFloat) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_csrSwapInd[idx]==d_csrSwapInd[idx+1] ) d_csrFlagFloat[idx]=(float)1;
        else d_csrFlagFloat[idx]=(float)0;
    }
}

void dense2csr( cusparseHandle_t handle, const int m, const cusparseMatDescr_t descr, const float *d_csrFlagFloat, const int *d_nnzPtr, float *d_csrFloat, int *d_csrRowUseless, int *d_csrFlagGood, CudaContext& context ) {
    cusparseSdense2csr( handle, 1, m, descr, d_csrFlagFloat, 1, d_nnzPtr, d_csrFloat, d_csrRowUseless, d_csrFlagGood );
}

void spsvBfs( const int vertex, const int edge, const int m, const int *h_csrRowPtr, const int *d_csrRowPtr, const int *d_csrColInd, int *d_bfsResult, const int depth, CudaContext& context ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr); 

    // h_csrVecInd - index to nonzero vector values
    // h_csrVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_csrVecCount - number of nonzero vector values
    int *h_csrVecInd;
    int *d_csrVecInd;
    int *h_csrVecVal;
    int *d_csrVecVal;
    int h_csrVecCount;
    int *d_csrSwapInd;

    h_csrVecInd = (int *)malloc(m*sizeof(int));
    h_csrVecInd[0] = vertex;
    h_csrVecInd[1] = 2;
    h_csrVecInd[2] = 3;
    h_csrVecVal = (int *)malloc(m*sizeof(int));
    h_csrVecVal[0] = 0; // Source node always defined as zero
    h_csrVecCount = 3;

    cudaMalloc(&d_csrVecInd, m*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrVecVal, m*sizeof(int));
    cudaMemcpy(d_csrVecVal, h_csrVecVal, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrSwapInd, m*sizeof(int));

    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff, *d_csrRowScan, *d_csrFlagGood, *d_csrRowUseless, *d_nnzPtr;
    cudaMalloc(&d_csrRowGood, m*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));
    cudaMalloc(&d_csrRowScan, m*sizeof(int));
    cudaMalloc(&d_csrFlagGood, m*sizeof(int));
    cudaMalloc(&d_csrRowUseless, m*sizeof(int));
    cudaMalloc(&d_nnzPtr, 2*sizeof(int));

    float *d_csrFlagFloat, *d_csrFloat;
    cudaMalloc(&d_csrFlagFloat, m*sizeof(float));
    cudaMalloc(&d_csrFloat, m*sizeof(float));

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    int *h_csrRowDiff = (int*)malloc(m*sizeof(int));
    float *h_csrFlagFloat = (float*)malloc(m*sizeof(float));
    int *h_nnzPtr = (int*)malloc(sizeof(int));

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;

    //print_array(h_csrRowPtr, 40);
    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d_csrRowDiff, m);
    gather( h_csrVecCount, d_csrVecInd, d_csrRowPtr, d_csrRowGood, context );
    gather( h_csrVecCount, d_csrVecInd, d_csrRowDiff, d_csrRowBad, context );
    int total = Reduce( d_csrRowBad, h_csrVecCount, context );
    Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, (int*)0, d_csrRowScan, context );
    IntervalGather( total, d_csrRowGood, d_csrRowScan, h_csrVecCount, d_csrColInd, d_csrSwapInd, context );
    MergesortKeys(d_csrSwapInd, total, mgpu::less<int>(), context);
    lookRight<<<NBLOCKS,NTHREADS>>>(d_csrSwapInd, total, d_csrFlagFloat);
    cusparseSnnz( handle, CUSPARSE_DIRECTION_ROW, 1, m, descr, d_csrFlagFloat, 1, d_nnzPtr, h_nnzPtr );
    dense2csr( handle, m, descr, d_csrFlagFloat, d_nnzPtr, d_csrFloat, d_csrRowUseless, d_csrFlagGood, context );
    BulkRemove( d_csrSwapInd, total, d_csrFlagGood, *h_nnzPtr, d_csrVecInd, context );
    printf("Removing %d entries from %d.\n", *h_nnzPtr, total);

    //updateBfs<<<NBLOCKS,NTHREADS>>>( d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, d_bfsResult );

    cudaMemcpy(h_csrRowDiff, d_csrVecInd, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrRowDiff, 40);
    //cudaMemcpy(h_csrRowDiff, d_csrFlagGood, m*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_csrRowDiff, 40);

    for( iter=2; iter<depth; iter++ ) {
    }

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );

    cudaFree(d_csrVecInd);
    cudaFree(d_csrVecVal);
    cudaFree(d_csrSwapInd);
    
    free(h_csrVecInd);
    free(h_csrVecVal);
}
