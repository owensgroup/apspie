// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include <cub/cub.cuh>

//#define NBLOCKS 16384
#define NTHREADS 512

__global__ void diff( const int *d_csrRowPtr, int *d_csrRowDiff, const int m ) {

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<m; idx+=blockDim.x*gridDim.x) {
        d_csrRowDiff[idx] = d_csrRowPtr[idx+1]-d_csrRowPtr[idx];
    }
}
 
__global__ void lookRight( const int *d_csrSwapInd, const int total, int *d_csrFlag) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_csrSwapInd[idx]!=d_csrSwapInd[idx+1] ) d_csrFlag[idx]=0;
        else d_csrFlag[idx]=1;
    }
}

__global__ void preprocessFlag( int *d_csrFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[idx] = 0;
}

__global__ void streamCompact( const int *d_csrFlag, const int *d_csrRowGood, int *d_csrVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_csrFlag[idx] ) d_csrVecInd[d_csrRowGood[idx]]=idx;
}

__global__ void scatter( const int total, const int *d_csrVecInd, int *d_csrFlag ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[d_csrVecInd[idx]] = 1;
}
        
template<typename typeVal>
__global__ void buildVector( const int m, const float minimum, const typeVal *d_randVec, const int *d_misResult, int *d_inputVector ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_misResult[idx]==-1 && d_randVec[idx] > minimum )
            d_inputVector[idx] = 1;
        else
            d_inputVector[idx] = 0;
    }
}

template<typename typeVal>
__global__ void updateMis( const int m, int *d_misResult, const typeVal *d_values, const typeVal *d_randVec, const int *d_inputVector) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_inputVector[idx]==1 && d_randVec[idx] > d_values[idx] )
            d_misResult[idx] = 1;
}

template< typename T >
void spmspvCsr( const T *d_inputVector, const int edge, const int m, const int *d_csrRowPtr, const int *d_csrColInd, T *d_spmspvResult, mgpu::CudaContext& context ) {
    int *h_csrVecInd = (int*)malloc(m*sizeof(int));

    int *d_csrVecInd, *d_csrSwapInd;
    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));

    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff;
    cudaMalloc(&d_csrRowGood, edge*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    int *h_csrRowDiff = (int*)malloc(m*sizeof(int));

    MGPU_MEM(int) ones = context.Fill( m, 1 );
    MGPU_MEM(int) index= context.FillAscending( m, 0, 1 );

    // Allocate device array
    cub::DoubleBuffer<int> d_keys(d_csrVecInd, d_csrSwapInd);

    int h_csrVecCount;
    int total;

    gpu_timer.Start();
    
    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d_csrRowDiff, m);
    mgpu::Scan<mgpu::MgpuScanTypeExc>( d_inputVector, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
    streamCompact<<<NBLOCKS,NTHREADS>>>( d_inputVector, d_csrRowGood, d_keys.Current(), m );

    // Gather from CSR graph into one big array
    IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
    mgpu::Scan<mgpu::MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
    IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
    IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_keys.Current(), context );

    //printf("The number of elements is %d\n", total);
    //cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_csrVecInd,40);

    // Reset dense flag array
    preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_spmspvResult, m );

    // Sort step
    //IntervalGather( ceil(h_csrVecCount/2.0), everyOther->get(), index->get(), ceil(h_csrVecCount/2.0), d_csrRowGood, d_csrRowBad, context );
    //SegSortKeysFromIndices( d_keys.Current(), total, d_csrRowBad, ceil(h_csrVecCount/2.0), context );
    //LocalitySortKeys( d_keys.Current(), total, context );
    //cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, d_keys, total );
    //MergesortKeys(d_keys.Current(), total, mgpu::less<int>(), context);

    // Scatter into dense flag array
    //IntervalScatter( total, d_keys.Current(), index_big->get(), total, ones_big->get(), d_spmspvResult, context );
    scatter<<<NBLOCKS,NTHREADS>>>( total, d_keys.Current(), d_spmspvResult );
    //cudaMemcpy(h_csrVecInd, d_spmspvResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_csrVecInd,40);

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);

    cudaFree(d_csrRowGood);
    cudaFree(d_csrRowBad);
    cudaFree(d_csrRowDiff);
}

template<typename typeVal>
void spmspvMis( const int edge, const int m, const int *h_csrRowPtr, const int *d_csrRowPtr, const int *d_csrColInd, const typeVal *d_randVec, int *d_misResult, const float delta, mgpu::CudaContext& context ) {

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
    typeVal *h_csrVecVal;
    typeVal *d_csrVecVal;
    typeVal *d_csrSwapVal;
    int h_csrVecCount;

    h_csrVecInd = (int *)malloc(edge*sizeof(int));
    h_csrVecVal = (typeVal *)malloc(edge*sizeof(typeVal));
    h_csrVecCount = 1;

    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrVecVal, edge*sizeof(typeVal));
    cudaMalloc(&d_csrSwapVal, edge*sizeof(typeVal));

    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff;
    cudaMalloc(&d_csrRowGood, edge*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    int *h_csrRowDiff = (int*)malloc(m*sizeof(int));
    int *d_inputVector;
    cudaMalloc(&d_inputVector, m*sizeof(int));

    int *h_misResult = (int*)malloc(m*sizeof(int));
    for( int i=0;i<m;i++ ) h_misResult[i] = -1;
    //h_misResult[1] = 1;
    //h_misResult[3] = 1;
    //h_misResult[4] = 1;
    cudaMemcpy(d_misResult, h_misResult, m*sizeof(int), cudaMemcpyHostToDevice); 

    MGPU_MEM(int) ones = context.Fill( m, 1 );
    MGPU_MEM(int) index= context.FillAscending( m, 0, 1 );
    //MGPU_MEM(int) ones_big = context.Fill( edge, 1 );
    //MGPU_MEM(int) index_big= context.FillAscending( edge, 0, 1 );
    MGPU_MEM(int) blockIndex = context.FillAscending( NBLOCKS, 0, NTHREADS );
    MGPU_MEM(int) everyOther = context.FillAscending( m, 0, 2 );

    // Allocate device array
    cub::DoubleBuffer<int> d_keys(d_csrVecInd, d_csrSwapInd);
    cub::DoubleBuffer<typeVal> d_vals(d_csrVecVal, d_csrSwapVal);

    // Allocate temporary storage
    size_t temp_storage_bytes = 93184;
    void *d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;
    int total= 0;
    cudaProfilerStart();

    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d_csrRowDiff, m);

    float minimum = 1;
    for( iter=1; iter<2; iter++ ) {
        
        // Update minimum threshold
        minimum -= delta;

        // Check which values are less than delta
        buildVector<<<NBLOCKS,NTHREADS>>>( m, minimum, d_randVec, d_misResult, d_inputVector );
        cudaMemcpy(h_csrVecInd, d_inputVector, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(h_csrVecInd,40);

        // Compact dense vector into sparse
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d_inputVector, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d_csrRowGood, context );
        streamCompact<<<NBLOCKS,NTHREADS>>>( d_inputVector, d_csrRowGood, d_keys.Current(), m );
        //IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_randVec, d_vals.Alternate(), context );

        // Gather from CSR graph into one big array              |     |  |
        // 1. Extracts the row lengths we are interested in e.g. 3  3  3  2  3  1
        // 2. Scans them, giving the offset from 0               0  3  6  8
        // 3. Extracts the row indices we are interested in      0  3  6  9 11 14 15
        // 4. Extracts the neighbour lists
        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowDiff, d_csrRowBad, context );
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d_keys.Current(), index->get(), h_csrVecCount, d_csrRowPtr, d_csrRowBad, context );
        IntervalExpand( total, d_csrRowGood, d_vals.Alternate(), h_csrVecCount, d_vals.Current(), context );
        TO-DO: //IntervalExpand( total, d_csrRowGood, d_keys.Current(), h_csrVecCount, d_vals.Current(), context );
        IntervalGather( total, d_csrRowBad, d_csrRowGood, h_csrVecCount, d_csrColInd, d_keys.Current(), context );

        cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(h_csrVecInd,40);
        cudaMemcpy(h_csrVecVal, d_vals.Current(), m*sizeof(typeVal), cudaMemcpyDeviceToHost);
        print_array(h_csrVecVal,40);

        // Reset dense flag array
        //preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_csrFlag, m );

        // Sort step
        //IntervalGather( ceil(h_csrVecCount/2.0), everyOther->get(), index->get(), ceil(h_csrVecCount/2.0), d_csrRowGood, d_csrRowBad, context );
        //SegSortKeysFromIndices( d_keys.Current(), total, d_csrRowBad, ceil(h_csrVecCount/2.0), context );
        //LocalitySortKeys( d_keys.Current(), total, context );
        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d_keys, d_vals, total );
        //cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, d_keys, total );
        //MergesortKeys(d_keys.Current(), total, mgpu::less<int>(), context);

        cudaMemcpy(h_csrVecInd, d_keys.Current(), m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(h_csrVecInd,40);
        cudaMemcpy(h_csrVecVal, d_vals.Current(), m*sizeof(typeVal), cudaMemcpyDeviceToHost);
        print_array(h_csrVecVal,40);

        // Segmented Reduce By Key
        //ReduceByKey( d_keys.Current(), d_vals.Current(), total, (float)0, mgpu::maximum<float>(), mgpu::equal_to<int>(), d_keys.Alternate(), d_vals.Alternate(), &h_csrVecCount, context );
        ReduceByKey( d_keys.Current(), d_vals.Current(), total, FLT_MIN, mgpu::maximum<float>(), mgpu::equal_to<int>(), d_keys.Alternate(), d_vals.Alternate(), &h_csrVecCount, (int*)0, context );

        cudaMemcpy(h_csrVecInd, d_keys.Alternate(), m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(h_csrVecInd,40);
        cudaMemcpy(h_csrVecVal, d_vals.Alternate(), m*sizeof(typeVal), cudaMemcpyDeviceToHost);
        print_array(h_csrVecVal,40);

        // Scatter into dense flag array
        //IntervalScatter( total, d_keys.Current(), index_big->get(), total, ones_big->get(), d_csrFlag, context );
        //scatter<<<NBLOCKS,NTHREADS>>>( total, d_keys.Current(), d_csrFlag );

        // Update MIS first, then update its neighbors
        updateMis<<<NBLOCKS,NTHREADS>>>( m, d_misResult, d_vals.Alternate(), d_randVec, d_inputVector);
        updateNeighbor<<<NBLOCKS,NTHREADS>>>( m, d_misResult, d_vals.Alternate
        cudaMemcpy(h_csrVecInd, d_misResult, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(h_csrVecInd,40);

//    printf("Running iteration %d.\n", iter);
//    gpu_timer.Stop();
//    elapsed = gpu_timer.ElapsedMillis();
//    printf("\nGPU BFS finished in %f msec. \n", elapsed);
//    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_csrVecCount, total);
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU MIS finished in %f msec. \n", elapsed);

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );

    cudaFree(d_csrRowGood);
    cudaFree(d_csrRowBad);
    cudaFree(d_csrRowDiff);
    cudaFree(d_inputVector);

    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    
    free(h_csrVecInd);
}

