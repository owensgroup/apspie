// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include <cub/cub.cuh>

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
    for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < length; idx += gridDim.x*blockDim.x) {
        if( d_spmvResult[idx]>0 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
        else d_spmvResult[idx] = 0;
    }
}

__global__ void updateBfsMerge( int *d_bfsResult, int *d_spmvResult, const int iter, const int length ) {
    for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < length; idx += gridDim.x*blockDim.x) {
        if( d_spmvResult[idx]>0 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
    }
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

__global__ void streamCompact( const int *d_csrFlag, const int *d_csrRowGood, int *d_csrVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_csrFlag[idx] ) d_csrVecInd[d_csrRowGood[idx]]=idx;
        //if( d_csrFlag[idx]==1 ) d_csrVecInd[d_csrRowGood[idx]]=idx;
}

__global__ void scatter( const int total, const int *d_csrVecInd, int *d_csrFlag ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[d_csrVecInd[idx]] = 1;
}

void spsvBfs( const int vertex, const int edge, const int m, const int *h_csrRowPtr, const int *d_csrRowPtr, const int *d_csrColInd, int *d_bfsResult, const int depth, const int sort, CudaContext& context ) {

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
    int *d_csrSwapInd;
    int h_csrVecCount;

    h_csrVecInd = (int *)malloc(edge*sizeof(int));

    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));
    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff, *d_nnzPtr;
    cudaMalloc(&d_csrRowGood, edge*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));
    cudaMalloc(&d_nnzPtr, 2*sizeof(int));

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    int *h_csrRowDiff = (int*)malloc(m*sizeof(int));
    float *h_csrFlagFloat = (float*)malloc(m*sizeof(float));
    int *h_nnzPtr = (int*)malloc(sizeof(int));
    int *d_csrFlag;
    cudaMalloc(&d_csrFlag, m*sizeof(int));

    int *h_bfsResult = (int*)malloc(m*sizeof(int));
    MGPU_MEM(int) ones = context.Fill( m, 1 );
    MGPU_MEM(int) index= context.FillAscending( m, 0, 1 );
    //MGPU_MEM(int) ones_big = context.Fill( edge, 1 );
    //MGPU_MEM(int) index_big= context.FillAscending( edge, 0, 1 );
    MGPU_MEM(int) blockIndex = context.FillAscending( NBLOCKS, 0, NTHREADS );

    // Allocate device array
    cub::DoubleBuffer<int> d_keys(d_csrVecInd, d_csrSwapInd);

    // Allocate temporary storage
    size_t temp_storage_bytes = 93184;
    void *d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for( int test=0;test<10;test++) {
    h_csrVecInd[0] = vertex;
    h_csrVecCount = 1;
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);
    for( int i=0;i<m;i++ ) h_bfsResult[i] = -1;
    h_bfsResult[vertex]=0;
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice); 

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;
    int flag = 0;
    int total= 0;
    int traversed = 0;
    //cudaProfilerStart();

    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d_csrRowDiff, m);

    for( iter=1; iter<depth; iter++ ) {

        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
        Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );

        preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_csrFlag, m );
        IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_keys.Current(), context );

//    cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);
        // Sort step
        switch(sort) {
            case(1):
        //SegSortKeysFromIndices( d_keys.Current(), total, blockIndex->get(), NBLOCKS, context );
        //LocalitySortKeys( d_keys.Current(), total, context );
                MergesortKeys(d_keys.Current(), total, mgpu::less<int>(), context);
                break;
            case(2):
                cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, d_keys, total );
                break;
        }
//    cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);

        //IntervalScatter( total, d_keys.Current(), index_big->get(), total, ones_big->get(), d_csrFlag, context );
        scatter<<<NBLOCKS,NTHREADS>>>( total, d_keys.Current(), d_csrFlag );

        updateBfs<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_csrFlag, iter, m );
        Scan<MgpuScanTypeExc>( d_csrFlag, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
        streamCompact<<<NBLOCKS,NTHREADS>>>( d_csrFlag, d_csrRowGood, d_keys.Current(), m );

//    printf("Running iteration %d.\n", iter);
//    gpu_timer.Stop();
//    elapsed = gpu_timer.ElapsedMillis();
//    printf("\nGPU BFS finished in %f msec. \n", elapsed);
//    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_csrVecCount, total);
//    cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);
//    cudaMemcpy(h_csrVecInd, d_temp_storage, m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);
        traversed+=total;
    }

    //cudaProfilerStop();
    gpu_timer.Stop();
    if( test==0 ) printf("Traversed edges: %d\n", traversed);
    elapsed += gpu_timer.ElapsedMillis();
    //printf("GPU BFS finished in %f msec.\n", elapsed);
    }
    printf("%f\n", elapsed/10);    

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );

    cudaFree(d_csrRowGood);
    cudaFree(d_csrRowBad);
    cudaFree(d_csrRowDiff);
    cudaFree(d_csrFlag);

    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    
    free(h_csrVecInd);
}

