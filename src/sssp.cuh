// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "spmspvSssp.cuh"

//#define NBLOCKS 16384
#define NTHREADS 512


template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

template<typename Value>
__global__ void addResultSssp( Value *d_ssspResult, Value *d_spmvResult, const int iter, const int length ) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        Value temp = fmin(d_spmvResult[idx],d_ssspResult[idx]);
        d_ssspResult[idx] = temp;
        d_spmvResult[idx] = temp;
        //d_ssspResult[idx] = fmin(d_spmvResult[idx],d_ssspResult[idx]);
    }
}

template< typename Value >
void sssp( const int vertex, const int edge, const int m, const Value *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, Value *d_ssspResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    Value *h_ssspResult;
    float *h_spmvResult;
    h_ssspResult = (Value*)malloc(m*sizeof(Value));
    h_spmvResult = (float*)malloc(m*sizeof(float));

    for( int i=0; i<m; i++ ) {
        h_ssspResult[i]=1.70141e+38;
        h_spmvResult[i]=1.70141e+38;
        if( i==vertex ) {
            h_ssspResult[i]=0;
            h_spmvResult[i]=0;
        }
    }
    cudaMemcpy(d_ssspResult, h_ssspResult, m*sizeof(Value), cudaMemcpyHostToDevice);
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
    //spmv<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
    spmspvSssp<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);

    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    //axpy(d_spmvSwap, d_bfsValA, m);
    addResultSssp<<<NBLOCKS,NTHREADS>>>( d_ssspResult, d_spmvResult, 1, m);

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<5; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            spmspvSssp<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            addResultSssp<<<NBLOCKS,NTHREADS>>>( d_ssspResult, d_spmvSwap, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            spmspvSssp<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            addResultSssp<<<NBLOCKS,NTHREADS>>>( d_ssspResult, d_spmvResult, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        }
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU SSSP finished in %f msec. \n", elapsed);
    //printf("The maximum frontier size was: %d.\n", frontier_max);
    //printf("The average frontier size was: %d.\n", frontier_sum/depth);

    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_ssspResult);
    free(h_spmvResult);
}
