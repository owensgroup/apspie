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

__global__ void spsv( const int *d_csrVecInd, const int *d_csrVecCount, const int *d_csrVecVal, const int *d_csrRowPtr, const int *d_csrColInd, const int edge, const int m, int *d_csrSwapInd, int *d_csrSwapCount, int *d_csrSwapVal, const int iter ) {
    const int STRIDE = gridDim.x * blockDim.x;
    const int BLOCKS = blockIdx.x * blockDim.x;
    //for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < *d_csrSwapCount; idx += STRIDE) {

    #pragma unroll
    for (int idx = 0; idx < *d_csrVecCount; idx++) {
        #pragma unroll
        for( int idy = threadIdx.x; idy < d_csrRowPtr[d_csrVecInd[idx]+1]; idy++ ) {
        //for( int idy = threadIdx.x; idy<2; idy++ ) {
            //int idy = threadIdx.x;
            d_csrSwapInd[idy] = d_csrColInd[d_csrRowPtr[d_csrVecInd[idx]]+idy];
            d_csrSwapVal[idy] = d_csrVecVal[d_csrRowPtr[d_csrVecInd[idx]]+idy];
        }
    }
}

void bulkExtract( const int *d_inputArray, const int h_inputCount, const int *h_csrRowPtr, int *d_outputArray, int h_outputCount, const int *h_csrVecInd, const int h_csrVecCount, int *d_swapArray, MGPU_MEM(int) d_csrBfsArray, CudaContext& context ) {
    int swapCount;
    
    for( int i=0; i<h_csrVecCount; i++ ) {
        swapCount = h_csrRowPtr[h_csrVecInd[i]+1]-h_csrRowPtr[h_csrVecInd[i]];
        mgpu::step_iterator<int> insertdata( h_csrRowPtr[h_csrVecInd[i]], 1 );
        //MGPU_MEM(int) insertdata = context.FillAscending(swapCount, h_csrRowPtr[h_csrVecInd[i]], 1);

        //PrintArray( insertdata->get(), 10, "%4d", 10 );
        printf("bulkExtract iteration %d: first is %d, last is %d\n", i, h_csrRowPtr[h_csrVecInd[i]], h_csrRowPtr[h_csrVecInd[i]+1]);

        /*if( i==0 ) BulkRemove( d_inputArray, h_inputCount, insertdata->get(), swapCount, d_outputArray, context );
        else {
            BulkRemove( d_inputArray, h_outputCount, insertdata->get(), swapCount, d_swapArray, context );
            SetOpKeys<MgpuSetOpUnion, true>(d_outputArray, h_outputCount, d_swapArray, swapCount, &d_csrBfsArray, context, false);
        }*/

        if( i==0 ) BulkRemove( d_inputArray, h_inputCount, insertdata, swapCount, d_outputArray, context );
        else {
            BulkRemove( d_inputArray, h_outputCount, insertdata, swapCount, d_swapArray, context );
            SetOpKeys<MgpuSetOpUnion, true>(d_outputArray, h_outputCount, d_swapArray, swapCount, &d_csrBfsArray, context, false);
        }

        h_outputCount = swapCount;
    }
}

void spsvBfs( const int vertex, const int edge, const int m, const int *h_csrRowPtr, const int *d_csrRowPtr, const int *d_csrColInd, int *d_bfsResult, const int depth, CudaContext& context ) {

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
    int *d_csrSwap2Ind;
    int *d_csrSwapVal;
    int h_csrSwapCount;

    h_csrVecInd = (int *)malloc(m*sizeof(int));
    h_csrVecInd[0] = vertex;
    h_csrVecInd[1] = 2;
    h_csrVecVal = (int *)malloc(m*sizeof(int));
    h_csrVecVal[0] = 0; // Source node always defined as zero
    h_csrVecCount = 2;
    h_csrSwapCount = 0;

    cudaMalloc(&d_csrVecInd, m*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csrVecVal, m*sizeof(int));
    cudaMemcpy(d_csrVecVal, h_csrVecVal, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csrSwapInd, m*sizeof(int));
    cudaMalloc(&d_csrSwap2Ind, m*sizeof(int));
    cudaMalloc(&d_csrSwapVal, m*sizeof(int));
    MGPU_MEM(int) intersectionDevice = context.Malloc<int>(m);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;

    //d_csrSwapCount[0] = d_csrRowPtr[d_csrVecInd[0]+1];
    //spsv<<<NBLOCKS,NTHREADS>>>( d_csrVecInd, d_csrVecCount, d_csrVecVal, d_csrRowPtr, d_csrColInd, edge, m, d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, iter );
    bulkExtract( d_csrColInd, m, h_csrRowPtr, d_csrSwapInd, h_csrSwapCount, h_csrVecInd, h_csrVecCount, d_csrVecInd, intersectionDevice, context );
    //updateBfs<<<NBLOCKS,NTHREADS>>>( d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, d_bfsResult );

    for( iter=2; iter<depth; iter++ ) {
        /*if( h_csrSwapCount%2==1 ) {
            cudaMemcpy( h_csrVecInd, d_csrSwapInd, h_csrSwapCount*sizeof(int), cudaMemcpyDeviceToHost );
        } else {
            cudaMemcpy( h_csrVecInd, d_csrVecInd, h_csrSwapCount*sizeof(int), cudaMemcpyDeviceToHost );
        }*/
        //spsv<<<NBLOCKS,NTHREADS>>>( d_csrVecInd, d_csrVecCount, d_csrVecVal, d_csrRowPtr, d_csrColInd, edge, m, d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, iter );
    }

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);

    cudaMemcpy(h_csrVecInd, d_csrVecInd, 40*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrVecInd, 40);
    PrintArray(*intersectionDevice, 40, "%4d", 10);

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtr, d_csrColInd, d_spsvResult );

    cudaFree(d_csrVecInd);
    cudaFree(d_csrVecVal);
    cudaFree(d_csrSwapInd);
    cudaFree(d_csrSwapVal);
    
    free(h_csrVecInd);
    free(h_csrVecVal);
}
