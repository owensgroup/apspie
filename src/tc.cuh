// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "mXv.cuh"
#include "scratch.hpp"

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

//template< typename T >
void allocScratch( d_scratch **d, const int edge, const int m ) {

    *d = (d_scratch *)malloc(sizeof(d_scratch));
    cudaMalloc(&((*d)->d_csrVecInd), edge*sizeof(int));
    cudaMalloc(&((*d)->d_csrSwapInd), edge*sizeof(int));
    cudaMalloc(&((*d)->d_csrVecVal), edge*sizeof(float));
    cudaMalloc(&((*d)->d_csrSwapVal), edge*sizeof(float));
    cudaMalloc(&((*d)->d_csrTempVal), edge*sizeof(float));

    cudaMalloc(&((*d)->d_csrRowGood), edge*sizeof(int));
    cudaMalloc(&((*d)->d_csrRowBad), m*sizeof(int));
    cudaMalloc(&((*d)->d_csrRowDiff), m*sizeof(int));
    cudaMalloc(&((*d)->d_ones), m*sizeof(int));
    cudaMalloc(&((*d)->d_index), m*sizeof(int));
    cudaMalloc(&((*d)->d_temp_storage), 93184);
    cudaMalloc(&((*d)->d_randVecInd), m*sizeof(int));

    //Host mallocs
    (*d)->h_csrVecInd = (int*) malloc (edge*sizeof(int));
    (*d)->h_csrVecVal = (float*) malloc (edge*sizeof(float));
    (*d)->h_csrRowDiff = (int*) malloc (m*sizeof(int));
    (*d)->h_ones = (int*) malloc (m*sizeof(int));
    (*d)->h_index = (int*) malloc (m*sizeof(int));

    (*d)->h_bfsResult = (int*) malloc (m*sizeof(int));
    (*d)->h_spmvResult = (float*) malloc (m*sizeof(float));
    (*d)->h_bfsValA = (float*) malloc (edge*sizeof(float));
}

template< typename T >
void bfs( const int vertex, const int edge, const int m, const T* d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    d_scratch *d;
    allocScratch( &d, edge, m );

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    // Generate d_ones, d_index
    for( int i=0; i<m; i++ ) {
        d->h_bfsResult[i]=-1;
        d->h_spmvResult[i]=0.0;
        d->h_ones[i] = 1;
        d->h_index[i] = i;
        if( i==vertex ) {
            d->h_bfsResult[i]=0;
            d->h_spmvResult[i]=1.0;
        }
    }
    cudaMemcpy(d_bfsResult, d->h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvSwap, d->h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_csrRowDiff
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_csrRowPtrA, d->d_csrRowDiff, m);

    // Generate values for BFS (csrValA where everything is 1)
    float *d_bfsValA;
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        d->h_bfsValA[i] = 1.0;
    }
    cudaMemcpy(d_bfsValA, d->h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();
    //spmv<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
    mXv<float>(d_spmvSwap, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, d, context);

    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1, m);

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<5; i++ ) {
        if( i%2==0 ) {
            //spmv<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, context);
            mXv<float>( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
            //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
            //print_array(h_spmvResult,m);
        } else {
            //spmv<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, context);
            mXv<float>( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult, d, context);
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i, m);
            //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
            //print_array(h_bfsResult,m);
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
}
