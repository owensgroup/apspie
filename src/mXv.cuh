// Provides BFS function for GPU

#include <fstream>
#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include "scratch.hpp"

#define NTHREADS 512

__global__ void diff( const int *d_cscColPtr, int *d_cscColDiff, const int m ) {

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<m; idx+=blockDim.x*gridDim.x) {
        d_cscColDiff[idx] = d_cscColPtr[idx+1]-d_cscColPtr[idx];
    }
}
 
__global__ void lookRight( const int *d_cscSwapInd, const int total, int *d_cscFlag) {
    
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<total; idx+=blockDim.x*gridDim.x) {
        if( d_cscSwapInd[idx]!=d_cscSwapInd[idx+1] ) d_cscFlag[idx]=0;
        else d_cscFlag[idx]=1;
    }
}

template<typename T>
__global__ void zeroArray( T *d_cscFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_cscFlag[idx] = 0;
}

template<typename T>
__global__ void zeroArrayMinPlus( T *d_csrFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_csrFlag[idx] = 1.70141e+38;
}

__global__ void bitify( const float *d_randVec, int *d_randVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_randVec[idx]>0.5 ) d_randVecInd[idx] = 1;
        else d_randVecInd[idx] = 0;
    }
}

__global__ void bitifyMinPlus( const float *d_randVec, int *d_randVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_randVec[idx]>-1 && d_randVec[idx]<1e38) d_randVecInd[idx] = 1;
        else d_randVecInd[idx] = 0;
    }
}

__global__ void streamCompact( const int *d_cscFlag, const int *d_cscColGood, int *d_cscVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_cscFlag[idx] ) d_cscVecInd[d_cscColGood[idx]]=idx;
}

__global__ void scatter( const int total, const int *d_cscVecInd, int *d_cscFlag ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_cscFlag[d_cscVecInd[idx]] = 1;
}

template<typename T>
__global__ void scatterFloat( const int total, const int *d_key, const T *d_cscSwapVal, T *d_temp ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) {
        d_temp[d_key[idx]] = d_cscSwapVal[idx];
        //printf("%d: \n", idx);
    }
}

__device__ void fatomicMin( float *addr, float val ) {
    if( *addr<=val ) return;
    int *const int_addr = (int*)addr;
    int old = *int_addr, temp;
    do {
        temp = old;
        if( __int_as_float(temp) <= val ) break;
        old = atomicCAS(int_addr, temp, __float_as_int(val));
    } while( old!=temp );
}

template<typename T>
__global__ void scatterAtomicAdd( const int total, const int *d_key, const T *d_cscSwapVal, T *d_temp ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        atomicAdd( &d_temp[d_key[idx]], d_cscSwapVal[idx] );
}

template<typename T>
__global__ void scatterAtomicMin( const int total, const int *d_key, const T *d_cscSwapVal, T *d_temp ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        fatomicMin( &d_temp[d_key[idx]], d_cscSwapVal[idx] );
}

template<typename T>
__global__ void gather( const int total, const int *d_cscVecInd, const T *d_randVec, T *d_cscVecVal ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        d_cscVecVal[idx] = d_randVec[d_cscVecInd[idx]];
}
        
template<typename T>
__global__ void buildVector( const int m, const float minimum, const T *d_randVec, const int *d_mmResult, int *d_inputVector ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_mmResult[idx]==-1 && d_randVec[idx] > minimum )
            d_inputVector[idx] = 1;
        else
            d_inputVector[idx] = 0;
    }
}

template<typename T>
__global__ void updateMis( const int m, int *d_mmResult, const T *d_cscSwapVal, const T *d_randVec, const int *d_inputVector) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_inputVector[idx]==1 && d_randVec[idx] > d_cscSwapVal[idx] )
            d_mmResult[idx] = 1;
}

template<typename T>
__global__ void updateNeighbor( const int total, int *d_mmResult, const int *d_key, const int *d_ind, const T *d_cscVecVal, const T *d_randVec ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) {
        int key = d_key[idx];
        int ind = d_ind[idx];
        if( d_cscVecVal[idx]==d_randVec[ind] && d_mmResult[ind]==1 )
            d_mmResult[key] = 0;
    }
}

// @brief ewiseMult for arithmetic mult semiring
template<typename T>
__global__ void ewiseMult( const int total, const T *d_x, const T*d_y, T *d_result ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) d_result[idx] = d_x[idx]*d_y[idx];
} 

// @brief ewiseMult for min-plus semiring
template<typename T>
__global__ void ewiseMultMinPlus( const int total, const T *d_x, const T*d_y, T *d_result ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x ) d_result[idx] = d_x[idx]+d_y[idx];
} 

__global__ void checkSorted( const int*d_csrVecInd, int *d_csrFlag, const int total ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total-1; idx+=blockDim.x*gridDim.x )
      if( d_csrVecInd[idx] > d_csrVecInd[idx+1] )
      {
        d_csrFlag[idx] = 1;
        //printf("Error: %d > %d not sorted\n", d_csrVecInd[idx], d_csrVecInd[idx+1] );
      } else d_csrFlag[idx] = 0;
}

__global__ void debugFilter( int total, const int *d_cscVecInd, int new_n, int *d_sum )
{
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<total; idx+=blockDim.x*gridDim.x )
        if( d_cscVecInd[idx] < new_n )
          atomicAdd( d_sum, 1 );
}

// @brief mXv for dense vector d_randVec
//
template<typename T>
int mXv( const T *d_randVec, const int edge, const int m, const T *d_cscVal, const int *d_cscColPtr, const int *d_cscRowInd, T *d_mmResult, d_scratch *d, const int op, mgpu::CudaContext& context ) {

    // h_cscVecInd - index to nonzero vector values
    // h_cscVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_cscVecCount - number of nonzero vector values
    int h_cscVecCount;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    size_t temp_storage_bytes = 93184;

    /*printf("randVec:\n");
    cudaMemcpy(d->h_cscVecVal, d_randVec, m*sizeof(float), cudaMemcpyDeviceToHost);
    printDevice(d->h_cscVecVal,10);
    printf("cscVal:\n");
    cudaMemcpy(d->h_cscVecVal, d_cscVal, m*sizeof(float), cudaMemcpyDeviceToHost);
    printDevice(d->h_cscVecVal,10);*/

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    //GpuTimer gpu_timer;
    //float elapsed = 0.0f;
    //gpu_timer.Start();
    int total= 0;
    //float minimum = 1;
    //cudaProfilerStart();

    //1. Obtain dense bit vector from dense vector
    //
    // op=1 Arithmetic semiring (any nonzero = 1)
    // op=2 MinPlus semiring (any value < 1e+38 = 1)
    if( op==1 ) bitify<<<NBLOCKS,NTHREADS>>>( d_randVec, d->d_randVecInd, m );
    else if( op==2 ) bitifyMinPlus<<<NBLOCKS,NTHREADS>>>( d_randVec, d->d_randVecInd, m );
     
    //2. Compact dense vector into sparse
    //    indices: d_cscColGood
    //     values: not necessary (will be expanded into d_cscVecVal in step 3
    mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_randVecInd, m, 0, mgpu::plus<int>(), (int*)0, &h_cscVecCount, d->d_cscColGood, context );
    if( h_cscVecCount == 0 ) {
        //printf( "Error: no frontier\n" );
        return 0;
    } else {
        streamCompact<<<NBLOCKS,NTHREADS>>>( d->d_randVecInd, d->d_cscColGood, d->d_cscVecInd, m );
        
        //3. Gather from CSR graph into one big array       |     |  |
        // 1. Extracts the row lengths we are interested in 3  3  3  2  3  1
        //  -> d_cscColBad
        // 2. Scans them, giving the offset from 0          0  3  6  8
        //  -> d_cscColGood
        // 3. Extracts the col indices we are interested in 0  6  9
        //  -> d_cscColBad
        // 4. Extracts the neighbour lists
        //  -> d_cscVecInd
        //  -> d_cscVecVal
        IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d->d_cscColDiff, d->d_cscColBad, context );
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_cscColBad, h_cscVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_cscColGood, context );
        IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d_cscColPtr, d->d_cscColBad, context );

        //printf("Processing %d nodes frontier size: %d\n", h_cscVecCount, total);

	// Vector Portion
        // a) naive method
        //   -IntervalExpand into frontier-length list
        //      1. Gather the elements indexed by d_cscVecInd
        //      2. Expand the elements to memory set by d_cscColGood
        //   -Element-wise multiplication with frontier
        IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d_randVec, d->d_cscTempVal, context );
        IntervalExpand( total, d->d_cscColGood, d->d_cscTempVal, h_cscVecCount, d->d_cscSwapVal, context );

        // Matrix Structure Portion
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscRowInd, d->d_cscVecInd, context );
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscVal, d->d_cscTempVal, context );

        /*printf("pre-ewiseMult:\n");
        cudaMemcpy(d->h_cscVecInd, d->d_cscVecInd, total*sizeof(int), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecInd,40);
        cudaMemcpy(d->h_cscVecVal, d->d_cscTempVal, total*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,40);*/

        // Element-wise multiplication
        //
        // op=1  arithmetic semiring
        // op=2  min-plus semiring
        if( op==1 ) ewiseMult<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );
	    else if( op==2 ) ewiseMultMinPlus<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );

        /*printf("post-ewiseMult:\n");
        cudaMemcpy(d->h_cscVecVal, d->d_cscVecVal, total*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,40);*/

        // Reset dense flag array
        //
        // op=1 Set all to Arithmetic semiring zero (0)
        // op=2 Set all to MinPlus semiring zero (1.70141e+38)
        if( op==1 ) zeroArray<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );
        else if( op==2 ) zeroArrayMinPlus<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

        // b) custom kernel method (fewer memory reads)
        // TODO
        if( op==1 ) scatterAtomicAdd<<<NBLOCKS,NTHREADS>>>( total, d->d_cscVecInd, d->d_cscVecVal, d_mmResult );
        else if( op==2 ) scatterAtomicMin<<<NBLOCKS,NTHREADS>>>( total, d->d_cscVecInd, d->d_cscVecVal, d_mmResult );
        /*

        //4. Sort step
        //IntervalGather( ceil(h_cscVecCount/2.0), everyOther->get(), d_index, ceil(h_cscVecCount/2.0), d_cscColGood, d_cscColBad, context );
        //SegSortKeysFromIndices( d_cscVecInd, total, d_cscColBad, ceil(h_cscVecCount/2.0), context );
        //LocalitySortKeys( d_cscVecInd, total, context );
        //cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );
        MergesortPairs(d->d_cscVecInd, d->d_cscVecVal, total, mgpu::less<int>(), context);

        printf("post-sort:\n");
        cudaMemcpy(d->h_cscVecInd, d->d_cscSwapInd, total*sizeof(int), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecInd,40);
        printf("post-sort:\n");
        cudaMemcpy(d->h_cscVecVal, d->d_cscSwapVal, total*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,40);*/

        //5. Gather the rand values
        //gather<<<NBLOCKS,NTHREADS>>>( total, d_cscVecVal, d_randVec, d_cscVecVal );

        //6. Segmented Reduce By Key
        //
        // op=1  arithmetic semiring
        // op=2  min-plus semiring
        //if( op==1 ) ReduceByKey( d->d_cscSwapInd, d->d_cscSwapVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d->d_cscVecInd, d->d_cscVecVal, &h_cscVecCount, (int*)0, context );
        //else if( op==2 ) ReduceByKey( d->d_cscSwapInd, d->d_cscSwapVal, total, (float)1.70141e+38, mgpu::minimum<float>(), mgpu::equal_to<int>(), d->d_cscVecInd, d->d_cscVecVal, &h_cscVecCount, (int*)0, context );
        if( op==1 ) ReduceByKey( d->d_cscVecInd, d->d_cscVecVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d->d_cscSwapInd, d->d_cscSwapVal, &h_cscVecCount, (int*)0, context );
        else if( op==2 ) ReduceByKey( d->d_cscVecInd, d->d_cscVecVal, total, (float)1.70141e+38, mgpu::minimum<float>(), mgpu::equal_to<int>(), d->d_cscSwapInd, d->d_cscSwapVal, &h_cscVecCount, (int*)0, context );

        /*printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);
        printf("post-reduce:\n");
        cudaMemcpy(d->h_cscVecInd, d->d_cscVecInd, total*sizeof(int), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecInd,40);
        printf("post-reduce:\n");
        cudaMemcpy(d->h_cscVecVal, d->d_cscVecVal, total*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,40);*/

        scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscSwapInd, d->d_cscSwapVal, d_mmResult );
        //scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscVecInd, d->d_cscVecVal, d_mmResult );

        // 8. Error checking. If misResult is all 0s, something has gone wrong.
        // Check using max reduce
        //mgpu::Reduce( d_mmResult, m, INT_MIN, mgpu::maximum<int>(), (int*)0, &total, context );
        //printf( "The biggest number in MIS result is %d\n", total );
        //if( total==0 )
        //    printf( "Error: no node generated\n" );
       /* printf("scatterFloat:\n"); 
        cudaMemcpy(d->h_cscVecInd, d->d_cscSwapInd, m*sizeof(int), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecInd,h_cscVecCount);
        cudaMemcpy(d->h_cscVecVal, d->d_cscSwapVal, m*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,h_cscVecCount);
        cudaMemcpy(d->h_cscVecVal, d_mmResult, m*sizeof(float), cudaMemcpyDeviceToHost);
        printDevice(d->h_cscVecVal,40);
    */
    
    //gpu_timer.Stop();
    //elapsed = gpu_timer.ElapsedMillis();
    //printf("GPU BFS finished in %f msec. \n", elapsed);
    //gpu_timer.Start();
    //printf("Keeping %d elements out of %d.\n", h_cscVecCount, total);
    return total;
    //    }
    //else if( minimum<=(float)0 )
    //    break;
    //}*/

    //cudaProfilerStop();
    //gpu_timer.Stop();
    //elapsed += gpu_timer.ElapsedMillis();
    //printf("\nGPU MM finished in %f msec. \n", elapsed);

    // 9. Error checking. If element of misResult is -1 something has gone wrong.
    // Check using min reduce
    //mgpu::Reduce( d_mmResult, m, INT_MAX, mgpu::minimum<int>(), (int*)0, &total, context );
    //printf( "The smallest number in MIS result is %d\n", total );
    //if( total==-1 )
    //    printf( "Error: MIS has -1 in it\n" );
    } 
}

// @brief mXv for sparse vector d_randVecInd, d_randVecVal
//
template<typename T>
int mXvSparse( const int *d_randVecInd, const T *d_randVecVal, const int edge, const int new_n, int m, int &nnz, const T *d_cscVal, const int *d_cscColPtr, const int *d_cscRowInd, int *d_resultInd, T *d_resultVal, d_scratch *d, mgpu::CudaContext &context) {

    // h_cscVecInd - index to nonzero vector values
    // h_cscVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_cscVecCount - number of nonzero vector values
    int h_cscVecCount;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    //GpuTimer gpu_timer;
    //float elapsed = 0.0f;
    //gpu_timer.Start();
    int flag = 0;
    int total= 0;
    //float minimum = 1;
    //cudaProfilerStart();

    // 1. We are given how many nonzeros exist in this column of B 
    h_cscVecCount = nnz;
        if( h_cscVecCount == 0 ) {
            //printf( "Error: no frontier\n" );
            return 0; 
		}
        
        //3. Gather from CSC graph into one big array       |     |  |
        // 1. Extracts the col lengths we are interested in 3  3  3  2  3  1
        //  -> d_cscColBad
        // 2. Scans them, giving the offset from 0          0  3  6  8
        //  -> d_cscColGood
        // 3. Extracts the col scans we are interested in   0  6  9
        //  -> d_cscColBad
        // 4. Extracts the neighbour lists
        //  -> d_cscVecInd
        //  -> d_cscVecVal
        IntervalGather( h_cscVecCount, d_randVecInd, d->d_index, h_cscVecCount, d->d_cscColDiff, d->d_cscColBad, context );
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_cscColBad, h_cscVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_cscColGood, context );
		if( total==0 ) {
            nnz = 0;
            //printf( "Error: dead-end node\n" );
            return 0; 
        }
			
        IntervalGather( h_cscVecCount, d_randVecInd, d->d_index, h_cscVecCount, d_cscColPtr, d->d_cscColBad, context );

	// Vector Portion
        // a) naive method
        //   -IntervalExpand into frontier-length list
        //      1. Gather the elements indexed by d_cscVecInd
        //      2. Expand the elements to memory set by d_cscColGood
        //   -Element-wise multiplication with frontier
        //IntervalGather( h_cscVecCount, d_cscVecInd, d->d_index, h_cscVecCount, d_randVec, d->d_cscTempVal, context );
        IntervalExpand( total, d->d_cscColGood, d_randVecVal, h_cscVecCount, d->d_cscSwapVal, context );

        // Matrix Structure Portion
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscRowInd, d->d_cscVecInd, context );
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscVal, d->d_cscTempVal, context );

        // Element-wise multiplication
        ewiseMult<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );

        // b) custom kernel method (fewer memory reads)
        // TODO
        
        // Reset dense flag array
        //preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

        //4. Sort step
        //IntervalGather( ceil(h_cscVecCount/2.0), everyOther->get(), d_index, ceil(h_cscVecCount/2.0), d_cscColGood, d_cscColBad, context );
        //SegSortKeysFromIndices( d_cscVecInd, total, d_cscColBad, ceil(h_cscVecCount/2.0), context );
        //LocalitySortKeys( d_cscVecInd, total, context );
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );

        cudaMalloc( &d_temp_storage, temp_storage_bytes );

        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );
        
        //MergesortPairs(d->d_cscVecInd, d->d_cscVecVal, total, mgpu::less<int>(), context);

        //5. Gather the rand values
        //gather<<<NBLOCKS,NTHREADS>>>( total, d_cscVecVal, d_randVec, d_cscVecVal );

        //6. Segmented Reduce By Key
        ReduceByKey( d->d_cscSwapInd, d->d_cscSwapVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d_resultInd, d_resultVal, &h_cscVecCount, (int*)0, context );
        //ReduceByKey( d->d_cscVecInd, d->d_cscVecVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d_resultInd, d_resultVal, &h_cscVecCount, (int*)0, context );

        //printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);

        nnz = h_cscVecCount;
        return total;
        //scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscSwapInd, d->d_cscSwapVal, d_mmResult );

        // 8. Error checking. If misResult is all 0s, something has gone wrong.
        // Check using max reduce
        //mgpu::Reduce( d_mmResult, m, INT_MIN, mgpu::maximum<int>(), (int*)0, &total, context );
        //printf( "The biggest number in MIS result is %d\n", total );
        //if( total==0 )
        //    printf( "Error: no node generated\n" );*/
    
//    printf("Running iteration %d.\n", iter);
//    gpu_timer.Stop();
//    elapsed = gpu_timer.ElapsedMillis();
//    printf("GPU BFS finished in %f msec. \n", elapsed);
//    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_cscVecCount, total);
    //    }
    //else if( minimum<=(float)0 )
    //    break;
    //}*/

    //cudaProfilerStop();
    //gpu_timer.Stop();
    //elapsed += gpu_timer.ElapsedMillis();
    //printf("\nGPU MM finished in %f msec. \n", elapsed);

    // 9. Error checking. If element of misResult is -1 something has gone wrong.
    // Check using min reduce
    //mgpu::Reduce( d_mmResult, m, INT_MAX, mgpu::minimum<int>(), (int*)0, &total, context );
    //printf( "The smallest number in MIS result is %d\n", total );
    //if( total==-1 )
    //    printf( "Error: MIS has -1 in it\n" );
}

// @brief mXv for sparse vector d_randVecInd, d_randVecVal
//
template<typename T>
int mXvSparseDebug( const int *d_randVecInd, const T *d_randVecVal, const int edge, const int new_n, int m, int &nnz, const T *d_cscVal, const int *d_cscColPtr, const int *d_cscRowInd, int *d_resultInd, T *d_resultVal, d_scratch *d, std::ofstream &outf, mgpu::CudaContext &context) {

    // h_cscVecInd - index to nonzero vector values
    // h_cscVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_cscVecCount - number of nonzero vector values
    int h_cscVecCount;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    //GpuTimer gpu_timer;
    //float elapsed = 0.0f;
    //gpu_timer.Start();
    int flag = 0;
    int total= 0;
    int h_sum = 0;
    int *d_sum;
    cudaMalloc( &d_sum, sizeof(int));
    cudaMemcpy( d_sum, &total, sizeof(int), cudaMemcpyHostToDevice );
    //float minimum = 1;
    //cudaProfilerStart();

    // 1. We are given how many nonzeros exist in this column of B 
    h_cscVecCount = nnz;
        if( h_cscVecCount == 0 ) {
            //printf( "Error: no frontier\n" );
            return 0; 
		}
		fprintDevice("randVec key", outf, d_randVecInd, nnz);
        fprintDevice("randVec Val", outf, d_randVecVal, nnz);
        
        //3. Gather from CSC graph into one big array       |     |  |
        // 1. Extracts the col lengths we are interested in 3  3  3  2  3  1
        //  -> d_cscColBad
        // 2. Scans them, giving the offset from 0          0  3  6  8
        //  -> d_cscColGood
        // 3. Extracts the col scans we are interested in   0  6  9
        //  -> d_cscColBad
        // 4. Extracts the neighbour lists
        //  -> d_cscVecInd
        //  -> d_cscVecVal
        IntervalGather( h_cscVecCount, d_randVecInd, d->d_index, h_cscVecCount, d->d_cscColDiff, d->d_cscColBad, context );
		fprintDevice("Col length", outf, d->d_cscColBad, h_cscVecCount);
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_cscColBad, h_cscVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_cscColGood, context );
		fprintDevice("Col length scan", outf, d->d_cscColGood, h_cscVecCount);
		if( total==0 ) {
            nnz = 0;
            //printf( "Error: dead-end node\n" );
            return 0; 
        }
			
        IntervalGather( h_cscVecCount, d_randVecInd, d->d_index, h_cscVecCount, d_cscColPtr, d->d_cscColBad, context );
		fprintDevice("Col length scan (good)", outf, d->d_cscColBad, h_cscVecCount);

        outf << "Processing " << h_cscVecCount << " nodes" << std::endl;
		outf << "Frontier size: " << total << std::endl;

	// Vector Portion
        // a) naive method
        //   -IntervalExpand into frontier-length list
        //      1. Gather the elements indexed by d_cscVecInd
        //      2. Expand the elements to memory set by d_cscColGood
        //   -Element-wise multiplication with frontier
        //IntervalGather( h_cscVecCount, d_cscVecInd, d->d_index, h_cscVecCount, d_randVec, d->d_cscTempVal, context );
        IntervalExpand( total, d->d_cscColGood, d_randVecVal, h_cscVecCount, d->d_cscSwapVal, context );

        // Matrix Structure Portion
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscRowInd, d->d_cscVecInd, context );
        IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscVal, d->d_cscTempVal, context );

        // Element-wise multiplication
        ewiseMult<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );
		fprintDevice("elementMul key", outf, d->d_cscVecInd, total);
        fprintDevice("elementMul Val", outf, d->d_cscVecVal,total);
		//fprintDeviceAll("elementMul key", outf, d->d_cscVecInd, total);

        debugFilter<<<NBLOCKS, NTHREADS>>>( total, d->d_cscVecInd, new_n, d_sum );
        cudaMemcpy( &h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost );
        outf << "The number of 0s in elementMul: " << h_sum << std::endl;
        
        // b) custom kernel method (fewer memory reads)
        // TODO
        
        // Reset dense flag array
        //preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

        //4. Sort step
        //IntervalGather( ceil(h_cscVecCount/2.0), everyOther->get(), d_index, ceil(h_cscVecCount/2.0), d_cscColGood, d_cscColBad, context );
        //SegSortKeysFromIndices( d_cscVecInd, total, d_cscColBad, ceil(h_cscVecCount/2.0), context );
        //LocalitySortKeys( d_cscVecInd, total, context );
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );

        cudaMalloc( &d_temp_storage, temp_storage_bytes );

        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );

		fprintDevice("In-loop SortPairs key", outf, d->d_cscVecInd, total);
        fprintDevice("In-loop SortPairs Val", outf, d->d_cscVecVal,total);
		//fprintDeviceAll("In-loop SortPairs key", outf, d->d_cscVecInd, total);

        checkSorted<<<NBLOCKS,NTHREADS>>>( d->d_cscSwapInd, d->d_cscVecInd, total );
        mgpu::Reduce( d->d_cscVecInd, total-1, (int)0, mgpu::plus<int>(), (int*)0, &flag, context );
        printf("The number of sort mistakes: %d out of %d\n", flag, total );
        //MergesortPairs(d->d_cscVecInd, d->d_cscVecVal, total, mgpu::less<int>(), context);

        //5. Gather the rand values
        //gather<<<NBLOCKS,NTHREADS>>>( total, d_cscVecVal, d_randVec, d_cscVecVal );

        //6. Segmented Reduce By Key
        //ReduceByKey( d->d_cscSwapInd, d->d_cscSwapVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d_resultInd, d_resultVal, &h_cscVecCount, (int*)0, context );
        ReduceByKey( d->d_cscVecInd, d->d_cscVecVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d_resultInd, d_resultVal, &h_cscVecCount, (int*)0, context );
		fprintDevice("ReduceByKey key", outf,d_resultInd, total);
        fprintDevice("ReduceByKey Val", outf,d_resultVal,total);

        //printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);

        nnz = h_cscVecCount;
        return total;
        //scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscSwapInd, d->d_cscSwapVal, d_mmResult );

        // 8. Error checking. If misResult is all 0s, something has gone wrong.
        // Check using max reduce
        //mgpu::Reduce( d_mmResult, m, INT_MIN, mgpu::maximum<int>(), (int*)0, &total, context );
        //printf( "The biggest number in MIS result is %d\n", total );
        //if( total==0 )
        //    printf( "Error: no node generated\n" );*/
    
//    printf("Running iteration %d.\n", iter);
//    gpu_timer.Stop();
//    elapsed = gpu_timer.ElapsedMillis();
//    printf("GPU BFS finished in %f msec. \n", elapsed);
//    gpu_timer.Start();
//    printf("Keeping %d elements out of %d.\n", h_cscVecCount, total);
    //    }
    //else if( minimum<=(float)0 )
    //    break;
    //}*/

    //cudaProfilerStop();
    //gpu_timer.Stop();
    //elapsed += gpu_timer.ElapsedMillis();
    //printf("\nGPU MM finished in %f msec. \n", elapsed);

    // 9. Error checking. If element of misResult is -1 something has gone wrong.
    // Check using min reduce
    //mgpu::Reduce( d_mmResult, m, INT_MAX, mgpu::minimum<int>(), (int*)0, &total, context );
    //printf( "The smallest number in MIS result is %d\n", total );
    //if( total==-1 )
    //    printf( "Error: MIS has -1 in it\n" );
}
