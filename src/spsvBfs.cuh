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

__global__ void updateBfs( int *d_bfsResult, int *d_spmvResult, const int iter, const int length ) {
    //const int STRIDE = gridDim.x * blockDim.x;
    //for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if( d_spmvResult[idx]>0 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
    //}
}

__global__ void updateBfsScatter( int *d_bfsResult, int *d_spmvResult, int *d_csrFlag, const int iter, const int length ) {
    //const int STRIDE = gridDim.x * blockDim.x;
    //for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if( d_spmvResult[idx]>0 && d_bfsResult[idx]<0 ) {
            d_bfsResult[idx] = iter;
            d_csrFlag[idx] = 1;
        } else d_csrFlag[idx] = 0;
    //}
}

__global__ void diff( const int *d_csrRowPtr, int *d_csrRowDiff, const int m ) {

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<m; idx+=blockDim.x*gridDim.x) {
        d_csrRowDiff[idx] = d_csrRowPtr[idx+1]-d_csrRowPtr[idx];
    }
}
 
__global__ void lookRightFloat( const int *d_csrSwapInd, const int total, float *d_csrFlagFloat) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_csrSwapInd[idx]==d_csrSwapInd[idx+1] ) d_csrFlagFloat[idx]=(float)1;
        else d_csrFlagFloat[idx]=(float)0;
    }
}

__global__ void lookRight( const int *d_csrSwapInd, const int total, int *d_csrFlag) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_csrSwapInd[idx]==d_csrSwapInd[idx+1] ) d_csrFlag[idx]=0;
        else d_csrFlag[idx]=1;
    }
}

void dense2csr( cusparseHandle_t handle, const int m, const cusparseMatDescr_t descr, const float *d_csrFlagFloat, const int *d_nnzPtr, float *d_csrFloat, int *d_csrRowUseless, int *d_csrFlagGood, CudaContext& context ) {
    cusparseSdense2csr( handle, 1, m, descr, d_csrFlagFloat, 1, d_nnzPtr, d_csrFloat, d_csrRowUseless, d_csrFlagGood );
}

__global__ void updateBfsSeq( const int *d_csrVecInd, const int h_csrVecCount, int *d_bfsResult, const int iter ) {
    for( int idx=0; idx<h_csrVecCount; idx++ ) {
        if( d_bfsResult[d_csrVecInd[idx]]<0 ) d_bfsResult[d_csrVecInd[idx]] = iter;
    }
}

__global__ void preprocessFlag( int *d_csrFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[idx] = 0;
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
    int h_csrVecCount;
    int *d_csrSwapInd;

    h_csrVecInd = (int *)malloc(m*sizeof(int));
    h_csrVecInd[0] = vertex;
    h_csrVecInd[1] = 2;
    h_csrVecInd[2] = 3;
    h_csrVecCount = 1;

    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));

    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff, *d_nnzPtr;
    cudaMalloc(&d_csrRowGood, edge*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));
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
    int *d_csrFlag;
    cudaMalloc(&d_csrFlag, edge*sizeof(int));

    int *h_bfsResult = (int*)malloc(m*sizeof(int));
    for( int i=0;i<m;i++ ) h_bfsResult[i] = -1;
    h_bfsResult[vertex]=0;
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice); 
    int *d_bfsSwap;
    cudaMalloc(&d_bfsSwap, m*sizeof(int));
    cudaMemcpy(d_bfsSwap, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    MGPU_MEM(int) ones = context.Fill( m, 1 );
    MGPU_MEM(int) index= context.FillAscending( m, 0, 1 );
    MGPU_MEM(int) ones_big = context.Fill( edge, 1 );
    MGPU_MEM(int) index_big= context.FillAscending( edge, 0, 1 );

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;
    int flag = 0;
    int total= 0;
    cudaProfilerStart();

    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d_csrRowDiff, m);

    for( iter=1; iter<depth; iter++ ) {

    if( flag==0 ) {
        IntervalGather( h_csrVecCount, d_csrVecInd, index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
        Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d_csrVecInd, index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
    } else {
        IntervalGather( h_csrVecCount, d_csrSwapInd, index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
        flag = 0;
        Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d_csrSwapInd, index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
    }
    printf("Running iteration %d.\n", iter);
    IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_csrVecInd, context );
    if( total>1 ) {
        /*MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), context);
        lookRight<<<NBLOCKS,NTHREADS>>>(d_csrVecInd, total, d_csrFlag);
        Scan<MgpuScanTypeExc>( d_csrFlag, total, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
        IntervalScatter( total, d_csrRowGood, index_big->get(), total, d_csrVecInd, d_csrSwapInd, context );
        IntervalScatter( h_csrVecCount, d_csrSwapInd, index_big->get(), h_csrVecCount, ones_big->get(), d_bfsSwap, context );*/

        preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_csrFlag, m );
        IntervalScatter( total, d_csrVecInd, index_big->get(), total, ones_big->get(), d_csrFlag, context );
        Scan<MgpuScanTypeExc>( d_csrFlag, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
        IntervalScatter( m, d_csrRowGood, index_big->get(), m, index_big->get(), d_csrSwapInd, context );
        IntervalScatter( h_csrVecCount, d_csrSwapInd, index_big->get(), h_csrVecCount, ones_big->get(), d_bfsSwap, context );
        flag = 1;
    } else {
        h_csrVecCount = 1;
        IntervalScatter( h_csrVecCount, d_csrVecInd, index->get(), h_csrVecCount, ones->get(), d_bfsSwap, context );
    }
    printf("Keeping %d elements out of %d.\n", h_csrVecCount, total);
    cudaMemcpy(h_csrVecInd, d_csrRowGood, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrVecInd,40);
    cudaMemcpy(h_csrVecInd, d_csrSwapInd, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrVecInd,40);
    cudaMemcpy(h_csrVecInd, d_csrFlag, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrVecInd,40);

    updateBfs<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_bfsSwap, iter, m );
    //updateBfsScatter<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_bfsSwap, d_csrFlag, iter, m );
    //Scan<MgpuScanTypeExc>( d_csrFlag, total, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );

    cudaFree(d_csrVecInd);
    cudaFree(d_csrSwapInd);
    cudaFree(d_csrRowGood);
    cudaFree(d_csrRowBad);
    cudaFree(d_csrRowDiff);
    cudaFree(d_csrFlagFloat);
    cudaFree(d_csrFloat);
    cudaFree(d_csrFlag);

    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    
    free(h_csrVecInd);
}
