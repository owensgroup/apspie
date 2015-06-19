// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include "spmspvMM.cuh"

//#define NBLOCKS 16384
#define NTHREADS 512

using namespace mgpu;

template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, CudaContext& context) {
    SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

__global__ void addResult( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        //d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
        if( d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0 ) d_bfsResult[idx] = iter;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

template< typename T >
void bfs( const int vertex, const int edge, const int m, const T* d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth, CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

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
            h_spmvResult[i]=i;
        }
    }
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvSwap, h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);

    // Generate values for BFS (csrValA where everything is 1)
    float *h_bfsValA, *d_bfsValA;
    h_bfsValA = (float*)malloc(edge*sizeof(float));
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        h_bfsValA[i] = 1;
    }
    cudaMemcpy(d_bfsValA, h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();
    //SpmvKernel<float>(d_spmvSwap, d_csrColIndA, d_csrRowPtrA, d_bfsValA, d_spmvResult, m, edge, context);
    //spmv<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
    spmspvMM<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);

    cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
    print_array(h_spmvResult,m);

    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1, m);
    cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_bfsResult,m);

    for( int i=2; i<=depth; i++ ) {
    //for( int i=2; i<3; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            spmspvMM<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i, m);
    cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            spmspvMM<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i, m);
    cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        }
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);
    //printf("The maximum frontier size was: %d.\n", frontier_max);
    //printf("The average frontier size was: %d.\n", frontier_sum/depth);

    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}
