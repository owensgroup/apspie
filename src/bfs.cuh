// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "mXv.cuh"
#include "scratch.hpp"

//#define NBLOCKS 16384
#define NTHREADS 512

template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

__global__ void addResult( int *d_bfsResult, float *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        //d_bfsResult[idx] = (d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0) ? iter:d_bfsResult[idx];
        if( d_spmvResult[idx]>0.5 && d_bfsResult[idx]<0 ) {
            d_bfsResult[idx] = iter;
        } else d_spmvResult[idx] = 0;
    }
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        //d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

template< typename T >
void bfs( const int vertex, const int edge, const int m, const T* d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, edge, m );

    // Allocate GPU memory for result
    int *h_bfsResult = (int*)malloc(m*sizeof(int));
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=-1;
        d->h_spmvResult[i]=0.0;
        d->h_ones[i] = 1;
        d->h_index[i] = i;
        if( i==vertex ) {
            h_bfsResult[i]=0;
            d->h_spmvResult[i]=1.0;
        }
    }
	//print_array(d->h_spmvResult, 40);
    cudaMemcpy(d_bfsResult, h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvSwap, d->h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, m);

    int cumsum = 0;
    int sum = 0;
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    for( int i=1; i<depth; i++ ) {
    //for( int i=2; i<5; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //cuspmv<float>( d_spmvResult, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, handle, descr);
            sum = mXv<float>( d_spmvResult, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i, m);
            
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, context);
            //cuspmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, handle, descr);
            sum = mXv<float>( d_spmvSwap, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        }
        cumsum+=sum;
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU BFS finished in %f msec. \n", elapsed);
    printf("Traversed edges: %d\n", cumsum);
    printf("Performance: %f GTEPS\n", (float)cumsum/(elapsed*1000000));

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
}
