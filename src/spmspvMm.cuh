// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include "scratch.hpp"

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

template<typename typeVal>
__global__ void preprocessFlag( typeVal *d_csrFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[idx] = 0;
}

__global__ void bitify( const float *d_randVec, int *d_randVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_randVec[idx]>0.5 ) d_randVecInd[idx] = 1;
        else d_randVecInd[idx] = 0;
    }
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
__global__ void scatterFloat( const int total, const int *d_key, const typeVal *d_csrSwapVal, typeVal *d_temp ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_temp[d_key[idx]] = d_csrSwapVal[idx];
}

template<typename typeVal>
__global__ void gather( const int total, const int *d_csrVecInd, const typeVal *d_randVec, typeVal *d_csrVecVal ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrVecVal[idx] = d_randVec[d_csrVecInd[idx]];
}
        
template<typename typeVal>
__global__ void buildVector( const int m, const float minimum, const typeVal *d_randVec, const int *d_mmResult, int *d_inputVector ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_mmResult[idx]==-1 && d_randVec[idx] > minimum )
            d_inputVector[idx] = 1;
        else
            d_inputVector[idx] = 0;
    }
}

template<typename typeVal>
__global__ void updateMis( const int m, int *d_mmResult, const typeVal *d_csrSwapVal, const typeVal *d_randVec, const int *d_inputVector) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_inputVector[idx]==1 && d_randVec[idx] > d_csrSwapVal[idx] )
            d_mmResult[idx] = 1;
}

template<typename typeVal>
__global__ void updateNeighbor( const int total, int *d_mmResult, const int *d_key, const int *d_ind, const typeVal *d_csrVecVal, const typeVal *d_randVec ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) {
        int key = d_key[idx];
        int ind = d_ind[idx];
        if( d_csrVecVal[idx]==d_randVec[ind] && d_mmResult[ind]==1 )
            d_mmResult[key] = 0;
    }
}
template<typename typeVal>
__global__ void elementMult( const int total, const typeVal *d_x, const typeVal*d_y, typeVal *d_result ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) d_result[idx] = d_x[idx]*d_y[idx];
} 

template<typename typeVal>
void spmspvMm( const typeVal *d_randVec, const int edge, const int m, const typeVal *d_csrVal, const int *d_csrRowPtr, const int *d_csrColInd, typeVal *d_mmResult, d_scratch *d, mgpu::CudaContext& context ) {

    // h_csrVecInd - index to nonzero vector values
    // h_csrVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_csrVecCount - number of nonzero vector values
    int h_csrVecCount;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    size_t temp_storage_bytes = 93184;

    printf("randVec:\n");
    cudaMemcpy(d->h_csrVecVal, d_randVec, m*sizeof(float), cudaMemcpyDeviceToHost);
    print_array(d->h_csrVecVal,10);

    /*int *h_csrVecInd;
    int *d_csrVecInd;
    int *d_csrSwapInd;
    typeVal *h_csrVecVal;
    typeVal *d_csrVecVal;
    typeVal *d_csrSwapVal;
    typeVal *d_csrTempVal;

    h_csrVecInd = (int *)malloc(edge*sizeof(int));
    h_csrVecVal = (typeVal *)malloc(edge*sizeof(typeVal));

    cudaMalloc(&d_csrVecInd, edge*sizeof(int));
    cudaMalloc(&d_csrSwapInd, edge*sizeof(int));
    cudaMalloc(&d_csrVecVal, edge*sizeof(typeVal));
    cudaMalloc(&d_csrSwapVal, edge*sizeof(typeVal));
    cudaMalloc(&d_csrTempVal, edge*sizeof(typeVal));

    int *d_csrRowGood, *d_csrRowBad, *d_csrRowDiff;
    cudaMalloc(&d_csrRowGood, edge*sizeof(int));
    cudaMalloc(&d_csrRowBad, m*sizeof(int));
    cudaMalloc(&d_csrRowDiff, m*sizeof(int));
    int *h_csrRowDiff = (int*)malloc(m*sizeof(int));

    int *h_ones = (int*)malloc(m*sizeof(int));
    int *h_index = (int*)malloc(m*sizeof(int));
    int *d_ones, *d_index;
    cudaMalloc(&d_ones, m*sizeof(int));
    cudaMalloc(&d_index, m*sizeof(int));

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Allocate for d_randVecInd
    int *d_randVecInd;
    cudaMalloc(&d_randVecInd, m*sizeof(int));*/

    // Initialize read-only arrays
    for( int i=0; i<m; i++ ) {
        d->h_ones[i] = 1;
        d->h_index[i] = i;
    }
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    //int iter = 0;
    int total= 0;
    //float minimum = 1;
    cudaProfilerStart();

    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtr, d->d_csrRowDiff, m);

    //1. Obtain dense bit vector from dense vector
    bitify<<<NBLOCKS,NTHREADS>>>( d_randVec, d->d_randVecInd, m );
     
    //2. Compact dense vector into sparse
    //    indices: d_csrRowGood
    //     values: not necessary (will be expanded into d_csrVecVal in step 3
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_randVecInd, m, 0, mgpu::plus<int>(), (int*)0, &h_csrVecCount, d->d_csrRowGood, context );
        if( h_csrVecCount == 0 ) 
            printf( "Error: no frontier\n" );
        else {
        streamCompact<<<NBLOCKS,NTHREADS>>>( d->d_randVecInd, d->d_csrRowGood, d->d_csrVecInd, m );
        printf("streamCompact:\n");
        cudaMemcpy(d->h_csrVecInd, d->d_csrVecInd, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,h_csrVecCount);
        
        //3. Gather from CSR graph into one big array       |     |  |
        // 1. Extracts the row lengths we are interested in 3  3  3  2  3  1
        //  -> d_csrRowBad
        // 2. Scans them, giving the offset from 0          0  3  6  8
        //  -> d_csrRowGood
        // 3. Extracts the col indices we are interested in 0  6  9
        //  -> d_csrRowBad
        // 4. Extracts the neighbour lists
        //  -> d_csrVecInd
        //  -> d_csrVecVal
        IntervalGather( h_csrVecCount, d->d_csrVecInd, d->d_index, h_csrVecCount, d->d_csrRowDiff, d->d_csrRowBad, context );
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_csrRowBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_csrRowGood, context );
        IntervalGather( h_csrVecCount, d->d_csrVecInd, d->d_index, h_csrVecCount, d_csrRowPtr, d->d_csrRowBad, context );

        //printf("Processing %d nodes frontier size: %d\n", h_csrVecCount, total);

	// Vector Portion
        // a) naive method
        //   -IntervalExpand into frontier-length list
        //      1. Gather the elements indexed by d_csrVecInd
        //      2. Expand the elements to memory set by d_csrRowGood
        //   -Element-wise multiplication with frontier
        IntervalGather( h_csrVecCount, d->d_csrVecInd, d->d_index, h_csrVecCount, d_randVec, d->d_csrTempVal, context );
        IntervalExpand( total, d->d_csrRowGood, d->d_csrTempVal, h_csrVecCount, d->d_csrSwapVal, context );

        // Matrix Structure Portion
        IntervalGather( total, d->d_csrRowBad, d->d_csrRowGood, h_csrVecCount, d_csrColInd, d->d_csrVecInd, context );
        IntervalGather( total, d->d_csrRowBad, d->d_csrRowGood, h_csrVecCount, d_csrVal, d->d_csrTempVal, context );
        printf("MatrixStructurePortion:\n");
        cudaMemcpy(d->h_csrVecInd, d->d_csrRowBad, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,total);
        cudaMemcpy(d->h_csrVecInd, d->d_csrRowGood, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,total);

        // Element-wise multiplication
        elementMult<<<NBLOCKS, NTHREADS>>>( total, d->d_csrSwapVal, d->d_csrTempVal, d->d_csrVecVal );
        printf("elementMult:\n"); 
        cudaMemcpy(d->h_csrVecInd, d->d_csrVecInd, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,total);
        cudaMemcpy(d->h_csrVecVal, d->d_csrSwapVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,total);
        cudaMemcpy(d->h_csrVecVal, d->d_csrTempVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,total);
        cudaMemcpy(d->h_csrVecVal, d->d_csrVecVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,total);

        // b) custom kernel method (fewer memory reads)
        // TODO
        
        // Reset dense flag array
        preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

        //4. Sort step
        //IntervalGather( ceil(h_csrVecCount/2.0), everyOther->get(), d_index, ceil(h_csrVecCount/2.0), d_csrRowGood, d_csrRowBad, context );
        //SegSortKeysFromIndices( d_csrVecInd, total, d_csrRowBad, ceil(h_csrVecCount/2.0), context );
        //LocalitySortKeys( d_csrVecInd, total, context );
        cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, d->d_csrVecInd, d->d_csrSwapInd, d->d_csrVecVal, d->d_csrSwapVal, total );
        //MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), context);

        //5. Gather the rand values
        //gather<<<NBLOCKS,NTHREADS>>>( total, d_csrVecVal, d_randVec, d_csrVecVal );

        //6. Segmented Reduce By Key
        ReduceByKey( d->d_csrVecInd, d->d_csrVecVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d->d_csrSwapInd, d->d_csrSwapVal, &h_csrVecCount, (int*)0, context );
        printf("ReduceByKey:\n"); 
        cudaMemcpy(d->h_csrVecInd, d->d_csrVecInd, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,total);
        cudaMemcpy(d->h_csrVecVal, d->d_csrVecVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,total);

        //printf("Current iteration: %d nonzero vector, %d edges\n",  h_csrVecCount, total);

        scatterFloat<<<NBLOCKS,NTHREADS>>>( h_csrVecCount, d->d_csrSwapInd, d->d_csrSwapVal, d_mmResult );

        // 8. Error checking. If misResult is all 0s, something has gone wrong.
        // Check using max reduce
        //mgpu::Reduce( d_mmResult, m, INT_MIN, mgpu::maximum<int>(), (int*)0, &total, context );
        //printf( "The biggest number in MIS result is %d\n", total );
        //if( total==0 )
        //    printf( "Error: no node generated\n" );*/
        printf("scatterFloat:\n"); 
        cudaMemcpy(d->h_csrVecInd, d->d_csrSwapInd, m*sizeof(int), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecInd,h_csrVecCount);
        cudaMemcpy(d->h_csrVecVal, d->d_csrSwapVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,h_csrVecCount);
        cudaMemcpy(d->h_csrVecVal, d_mmResult, m*sizeof(float), cudaMemcpyDeviceToHost);
        print_array(d->h_csrVecVal,40);
    
//    printf("Running iteration %d.\n", iter);
    gpu_timer.Stop();
    elapsed = gpu_timer.ElapsedMillis();
    printf("GPU BFS finished in %f msec. \n", elapsed);
    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_csrVecCount, total);
    //    }
    //else if( minimum<=(float)0 )
    //    break;
    //}*/

    //cudaProfilerStop();
    //gpu_timer.Stop();
    //elapsed += gpu_timer.ElapsedMillis();
    //printf("\nGPU MM finished in %f msec. \n", elapsed);

    // 9. Error checking. If element of misResult is -1 something has gone wrong.
    // CHeck using min reduce
    //mgpu::Reduce( d_mmResult, m, INT_MAX, mgpu::minimum<int>(), (int*)0, &total, context );
    //printf( "The smallest number in MIS result is %d\n", total );
    //if( total==-1 )
    //    printf( "Error: MIS has -1 in it\n" );

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );
    } 
}

