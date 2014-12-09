// Puts everything together
// For now, just run V times.
// Optimizations: 
// -come up with good stopping criteria
// -start from i=1
// -test whether float really are faster than ints

#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusparse.h>

#include <thrust/device_vector.h>
#include <cublas_v2.h>

#define N (33 * 1024)

template<typename T>
void print_vector( T *vector, int length ) {
    for( int j=0;j<length;j++ ) {
        std::cout << vector[j] << " " << j << "\n";
    }
}

template<typename T>
void print_array( T *array, int length ) {
    //if( length>50 ) length=50;
    for( int j=0;j<length;j++ ) {
        std::cout << array[j] << " ";
        if( j%20==19 )
            std::cout << "\n";
    }
    std::cout << "\n";
}

void coo2csr( const int *d_cooRowIndA, const int edge, const int m, int *d_csrRowPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("COO -> CSR conversion successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid value for idxbase.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
}

void csr2csc( const int m, const int edge, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_cscValA, int *d_cscRowIndA, int *d_cscColPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // For CUDA 4.0
    cusparseStatus_t status = cusparseScsr2csc(handle, m, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, 1, CUSPARSE_INDEX_BASE_ZERO);

    // For CUDA 5.0+
    //cusparseScsr2csc(handle, m, m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, 1, CUSPARSE_INDEX_BASE_ZERO);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("Transpose successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
}

void axpy( float *d_bfsResult, const float *d_csrValA, const int m ) {
    const float alf = -1;
    const float *alpha = &alf;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasSaxpy(handle, 
                            m, 
                            alpha, 
                            d_csrValA, 
                            1, 
                            d_bfsResult, 
                            1);

    switch( status ) {
        case CUBLAS_STATUS_SUCCESS:
	    printf("axpy completed successfully!\n");
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
	    printf("The library was not initialized.\n");
	    break;
        case CUBLAS_STATUS_ARCH_MISMATCH:	
	    printf("The device does not support double-precision.\n");
	    break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
	    printf("The function failed to launch on the GPU.\n");
    }

    cublasDestroy(handle);
}

void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult ) {
    const float alpha = 1;
    const float beta = 0;
    const float *value = &alpha;

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

    cusparseStatus_t status = cusparseScsrmv(handle, 
                   CUSPARSE_OPERATION_NON_TRANSPOSE, 
                   m, m, alpha, 
                   descr, 
                   //value,
                   d_csrValA, 
                   d_csrRowPtrA, 
                   d_csrColIndA, 
                   d_inputVector, 
                   beta,
                   d_spmvResult);

    // For CUDA 5.0+
    //cusparseStatus_t status = cusparseScsrmv(handle,                   CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, m, edge, alpha, descr, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_inputVector, m, beta, d_bfsResult, m);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("spmv multiplication successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("Error: Matrix type not supported.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

__global__ void addResult( float *d_bfsResult, const float *d_spmvResult, const int iter ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while( tid < N ) {
        d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void bfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_bfsResult ) {

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    float *h_bfsResult;
    h_bfsResult = (float*)malloc(m*sizeof(float));

    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=0;
        if( i==vertex )
            h_bfsResult[i]=1;
    }
    //std::cout << "This is m: " << m << std::endl;
    //print_array(h_bfsResult,m);
    cudaMemcpy(d_bfsResult,h_bfsResult, m*sizeof(float), cudaMemcpyHostToDevice);

    // Generate values for BFS (csrValA where everything is 1)
    float *h_bfsValA, *d_bfsValA;
    h_bfsValA = (float*)malloc(edge*sizeof(float));
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        h_bfsValA[i] = 1;
    }
    cudaMemcpy(d_bfsValA, h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);
    
    // Use Thrust to generate initial vector
    //thrust::device_ptr<float> dev_ptr(d_inputVector);
    //thrust::fill(dev_ptr, dev_ptr + m, (float) 0);
    //thrust::device_vector<float> inputVector(m, 0);
    //inputVector[vertex] = 1;
    //d_inputVector = thrust::raw_pointer_cast( &inputVector[0] );

    spmv(d_bfsResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult);
    
    axpy(d_bfsResult, d_bfsValA, m);
    addResult<<<128,128>>>( d_bfsResult, d_spmvResult, 1 );

    //for( int i=1; i<m-1; i++ ) {
    for( int i=2; i<3; i++ ) {
        if( i%2==0 ) {
            spmv( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap );
            addResult<<<128,128>>>( d_bfsResult, d_spmvSwap, i );
        } else {
            spmv( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult );
            addResult<<<128,128>>>( d_bfsResult, d_spmvResult, i );
        }
    }

    float *h_spmvResult;
    h_spmvResult = (float*)malloc(m*sizeof(float));
    cudaMemcpy(h_spmvResult,d_spmvResult, m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(float), cudaMemcpyDeviceToHost);
    //print_array(h_spmvResult,m);
    print_array(h_bfsResult,m);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}

int main(int argc, char**argv) {
    int m, n, edge;

    //for( int i=1; i<argc; i++ ) {
    //bfs(argv[i]);
    freopen(argv[1],"r",stdin);
    //freopen("log","w",stdout);

    int c = getchar();
    int old_c = 0;
    //printf("%d\n",c);
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, stdin);
            printf("%d %d\n",old_c,c);
            break;
        }
        old_c = c;
        c=getchar();
    }
    scanf("%d %d %d", &m, &n, &edge);
    
    // Allocate memory depending on how many edges are present
    float *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;

    h_csrValA    = (float*)malloc(edge*sizeof(float));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_csrColIndA[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowIndA[j]);

        if( j==0 ) {
            c=getchar();
            //printf("c = %d\n",c);
            //ungetc(c, stdin);
        }

        if( c!=32 ) {
            h_csrValA[j]=1.0;
        } else {
            scanf("%f", &h_csrValA[j]);
        }

        h_cooRowIndA[j]--;
        h_csrColIndA[j]--;
        //printf("%d %d %d\n", h_cooRowIndA[j], h_csrColIndA[j], j);
    }
    //print_array(h_csrValA,edge);

    // Allocate GPU memory
    float *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    float *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;

    cudaMalloc(&d_csrValA, edge*sizeof(float));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    cudaMalloc(&d_cscValA, edge*sizeof(float));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(h_cooRowIndA, d_cooRowIndA, (edge)*sizeof(int),cudaMemcpyDeviceToHost);
    //print_vector(h_cooRowIndA,edge);

    // Run COO -> CSR kernel
    coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // Run CSR -> CSC kernel
    csr2csc( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );

    // Run BFS kernel
    float *d_bfsResult;
    cudaMalloc(&d_bfsResult, (m+1)*sizeof(float));
    
    // Non-transpose spmv
    //bfs( i, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult );

    //bfs( 0, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_bfsResult );
    bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult );

    //for( int i=0;i<m;i++ ) {
    //    std::cout << i << " ";
    //
    //}

    // Copy data back to host
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    //print_array(h_csrRowPtrA,m+1);

    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_cooRowIndA);

    cudaFree(d_cscValA);
    cudaFree(d_cscRowIndA);
    cudaFree(d_cscColPtrA);
    cudaFree(d_bfsResult);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooRowIndA);

    //free(h_cscValA);
    //free(h_cscRowIndA);
    //free(h_cscColPtrA);
}
