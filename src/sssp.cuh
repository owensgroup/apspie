// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
//#include "spmspvSssp.cuh"
#include "scratch.hpp"
#include "mXv.cuh"

//#define NBLOCKS 16384
#define NTHREADS 512


template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_cscValA, d_cscRowIndA, edge, d_cscColPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

/*template<typename T>
void spmvMinPlus( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)9999.0, mgpu::plus<T>(), mgpu::minimum<T>(), context);
}*/

template<typename T>
__global__ void addResultSssp( T *d_ssspResult, T *d_spmvResult, T *d_spmvSwap, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        T temp = fmin(d_spmvResult[idx],d_ssspResult[idx]);
        d_spmvSwap[idx] -= temp;
        d_ssspResult[idx] = temp;
        d_spmvResult[idx] = temp;
        //d_ssspResult[idx] = fmin(d_spmvResult[idx],d_ssspResult[idx]);
    }
}

template< typename T >
void sssp( const int vertex, const int edge, const int m, const T *d_cscValA, const int *d_cscColPtrA, const int *d_cscRowIndA, T *d_ssspResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, edge, m );

    // Allocate GPU memory for result
    T *h_ssspResult = (T*) malloc( m*sizeof(T));
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        h_ssspResult[i]=1.7e+38;
        d->h_spmvResult[i]=1.7e+38;
        d->h_ones[i] = 1;
        d->h_index[i] = i;
        if( i==vertex ) {
            h_ssspResult[i]=0.0;
            d->h_spmvResult[i]=0.0;
        }
    }
    cudaMemcpy(d_ssspResult, h_ssspResult, m*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvSwap, d->h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, m);

    // Keep a count of edges traversed
    int cumsum = 0;
    int sum = 0;
    float change = 0;
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

    //for( int i=1; i<depth; i++ ) {
    for( int i=1; i<100; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //spmvMinPlus<float>( d_spmvResult, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);

            // op=2 MinPlus semiring
            sum = mXv<float>( d_spmvResult, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, d, 2, context);
            addResultSssp<<<NBLOCKS,NTHREADS>>>( d_ssspResult, d_spmvSwap, d_spmvResult,i, m);
            mgpu::Reduce( d_spmvResult, m, (float)0, mgpu::plus<float>(), (float*)0, &change, context );
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(d->h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //printf("spmvSwap:\n");
            //print_array(d->h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, context);
            //spmvMinPlus<float>( d_spmvSwap, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, context);

            // op=2 MinPlus semiring
            sum = mXv<float>( d_spmvSwap, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvResult, d, 2, context);
            addResultSssp<<<NBLOCKS,NTHREADS>>>( d_ssspResult, d_spmvResult, d_spmvSwap, i, m);
            mgpu::Reduce( d_spmvSwap, m, (float)0, mgpu::plus<float>(), (float*)0, &change, context );
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(d->h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            //printf("spmvResult:\n");
            //print_array(d->h_spmvResult,m);
        }
        //printf("Change in iter %d: %f\n", i, change);
        if( change<1.0 ) break;
        cumsum+=sum;
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU SSSP finished in %f msec. \n", elapsed);
    printf("Traversed edges: %d\n", cumsum);
    printf("Performance: %f GTEPS\n", (float)cumsum/(elapsed*1000000));
    //printf("The maximum frontier size was: %d.\n", frontier_max);
    //printf("The average frontier size was: %d.\n", frontier_sum/depth);

    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
}
