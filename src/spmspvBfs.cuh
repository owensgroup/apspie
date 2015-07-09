// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include <cub/cub.cuh>

//#define NBLOCKS 16384
#define NTHREADS 512

__global__ void updateBfs( int *d_bfsResult, int *d_spmvResult, const int iter, const int length ) {
    for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < length; idx += gridDim.x*blockDim.x) {
        if( d_spmvResult[idx]>0 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
        else d_spmvResult[idx] = 0;
    }
}

__global__ void updateSparseBfs( int *d_bfsResult, const int *d_csrVecInd, const int iter, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) {
        int index = d_csrVecInd[idx];
        if( d_bfsResult[index] == -1 ) d_bfsResult[index] = iter;
        //else d_csrVecInd[idx] = 0;
    }
}

__global__ void lookRightFloat( const int *d_csrSwapInd, const int total, float *d_csrFlagFloat) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_csrSwapInd[idx]==d_csrSwapInd[idx+1] ) d_csrFlagFloat[idx]=(float)1;
        else d_csrFlagFloat[idx]=(float)0;
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

/*template< typename T >
void spmspvCsr( const int *d_csrColInd, const int edge, const int *d_csrRowPtr, const int m, const T *d_inputVector, T *d_spmvResult, CudaContext& context ) {

    int h_csrVecCount;

    Scan<MgpuScanTypeExc>( d_csrFlag, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
    streamCompact<<<NBLOCKS,NTHREADS>>>( d_csrFlag, d_csrRowGood, d_inputVector, m );

    // Gather from CSR graph into one big array
    IntervalGather( h_csrVecCount, d_inputVector, index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
    Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
    IntervalGather( h_csrVecCount, d_inputVector, index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
    IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_keys.Current(), context );

    // Reset dense flag array
    preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_csrFlag, m );

    // Sort step
    //IntervalGather( ceil(h_csrVecCount/2.0), everyOther->get(), index->get(), ceil(h_csrVecCount/2.0), d_csrRowGood, d_csrRowBad, context );
    //SegSortKeysFromIndices( d_keys.Current(), total, d_csrRowBad, ceil(h_csrVecCount/2.0), context );
    //LocalitySortKeys( d_keys.Current(), total, context );
    //cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, d_keys, total );
    //MergesortKeys(d_keys.Current(), total, mgpu::less<int>(), context);

    //updateSparseBfs<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_keys.Current(), iter, total );

    // Scatter into dense flag array
    //IntervalScatter( total, d_keys.Current(), index_big->get(), total, ones_big->get(), d_csrFlag, context );
    scatter<<<NBLOCKS,NTHREADS>>>( total, d_keys.Current(), d_csrFlag );
}*/

void spmspvBfs( const int vertex, const int edge, const int m, const int *h_csrRowPtr, const int *d_csrRowPtr, const int *d_csrColInd, int *d_bfsResult, const int depth, CudaContext& context ) {

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
    h_csrVecInd[0] = vertex;
    h_csrVecCount = 1;

    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

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
    for( int i=0;i<m;i++ ) h_bfsResult[i] = -1;
    h_bfsResult[vertex]=0;
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice); 
    MGPU_MEM(int) ones = context.Fill( m, 1 );
    MGPU_MEM(int) index= context.FillAscending( m, 0, 1 );
    //MGPU_MEM(int) ones_big = context.Fill( edge, 1 );
    //MGPU_MEM(int) index_big= context.FillAscending( edge, 0, 1 );
    MGPU_MEM(int) blockIndex = context.FillAscending( NBLOCKS, 0, NTHREADS );
    MGPU_MEM(int) everyOther = context.FillAscending( m, 0, 2 );

    // Allocate device array
    cub::DoubleBuffer<int> d_keys(d_csrVecInd, d_csrSwapInd);
    //cub::DoubleBuffer<int> d_keys;
    //cudaMalloc(&d_keys.d_buffers[0], edge*sizeof(int));
    //cudaMalloc(&d_keys.d_buffers[1], edge*sizeof(int));
    //cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

    // Allocate temporary storage
    size_t temp_storage_bytes = 93184;
    void *d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

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

        // Compact dense vector into sparse
        if( iter>1 ) {
            Scan<MgpuScanTypeExc>( d_csrFlag, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
            streamCompact<<<NBLOCKS,NTHREADS>>>( d_csrFlag, d_csrRowGood, d_keys.Current(), m );
        }

        // Gather from CSR graph into one big array
        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
        Scan<MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
        IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_keys.Current(), context );

        // Reset dense flag array
        preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_csrFlag, m );

        // Sort step
        //IntervalGather( ceil(h_csrVecCount/2.0), everyOther->get(), index->get(), ceil(h_csrVecCount/2.0), d_csrRowGood, d_csrRowBad, context );
        //SegSortKeysFromIndices( d_keys.Current(), total, d_csrRowBad, ceil(h_csrVecCount/2.0), context );
        //LocalitySortKeys( d_keys.Current(), total, context );
        cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, d_keys, total );
        //MergesortKeys(d_keys.Current(), total, mgpu::less<int>(), context);

        //updateSparseBfs<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_keys.Current(), iter, total );

        // Scatter into dense flag array
        //IntervalScatter( total, d_keys.Current(), index_big->get(), total, ones_big->get(), d_csrFlag, context );
        scatter<<<NBLOCKS,NTHREADS>>>( total, d_keys.Current(), d_csrFlag );

        updateBfs<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_csrFlag, iter, m );

//    printf("Running iteration %d.\n", iter);
    gpu_timer.Stop();
    elapsed = gpu_timer.ElapsedMillis();
    printf("GPU BFS finished in %f msec. \n", elapsed);
    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_csrVecCount, total);
//    cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);
//    cudaMemcpy(h_csrVecInd, d_temp_storage, m*sizeof(int), cudaMemcpyDeviceToHost);
//    print_array(h_csrVecInd,40);
    }

//    cudaProfilerStop();
//    gpu_timer.Stop();
//    elapsed += gpu_timer.ElapsedMillis();
//    printf("\nGPU BFS finished in %f msec. \n", elapsed);

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

