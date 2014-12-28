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

__global__ void csrAddResult( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

void csrBfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth, CudaContext& context) {

    // h_csrVecInd - index to nonzero vector values
    // h_csrVecVal - for BFS, number of jumps from source
    //             - for SSSP, distance from source
    // h_csrVecCount - number of nonzero vector values
    int *h_csrVecInd;
    int *d_csrVecInd;
    int *h_csrVecVal;
    int *d_csrVecVal;
    int h_csrVecCount;
    int *d_csrVecCount;

    h_csrVecInd = (int *)malloc(m*sizeof(int));
    h_csrVecInd[0] = vertex;
    h_csrVecVal = (int *)malloc(m*sizeof(int));
    h_csrVecVal[0] = 0; // Source node always defined as zero
    h_csrVecCount = 1;

    cudaMalloc(&d_csrVecInd, m*sizeof(int));
    cudaMemcpy(d_csrVecInd, h_csrVecInd, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csrVecVal, m*sizeof(int));
    cudaMemcpy(d_csrVecVal, h_csrVecVal, h_csrVecCount*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_csrVecCount, sizeof(int));
    cudaMemcpy(d_csrVecCount, &h_csrVecCount, sizeof(int), cudaMemcpyHostToDevice);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    // First iteration
    // Note that updateBFS is similar to addResult kernel
    //   -has additional pruning function. If new node, keep. Otherwise, prune.
    gpu_timer.Start();
    int iter = 1;
    //spsv<<<NBLOCKS,NTHREADS>>>( d_csrVecInd, d_csrVecCount, d_csrRowPtrA, d_csrColIndA, edge, m, d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, iter );
    //updateBfs<<<NBLOCKS,NTHREADS>>>( d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, d_bfsResult );

    for( iter=2; iter<depth; iter++ ) {
        //spsv<<<NBLOCKS,NTHREADS>>>( d_csrVecInd, d_csrVecCount, d_csrRowPtrA, d_csrColIndA, edge, m, d_csrSwapInd, d_csrSwapCount, d_csrSwapVal, iter );
    }

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);

    // For future sssp
    //ssspSv( d_csrVecInd, edge, m, d_csrVal, d_csrRowPtrA, d_csrColIndA, d_spsvResult );
}
