// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include "mXv.cuh"
#include "scratch.hpp"
#include "matrix.hpp"
//#include "spgemmKernel.cuh"
#include "spgemmKernel2.cuh"
#include "common.hpp"
#include "triple.hpp"

#include "bhsparse.h"

#define NTHREADS 512
#define MAX_SHARED 49152
#define MAX_GLOBAL 270000000

int bhsparseSpgemm( d_matrix*, d_matrix*, d_matrix* );

// Matrix multiplication (Host code)
void spgemm( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {

	d_matrix Bsub, Blast;
	d_matrix *Cpiece = (d_matrix*) malloc(partNum*sizeof(d_matrix));
	matrix_new( &Bsub, B->m, partSize );
	matrix_new( &Blast,B->m, B->m-partSize*(partNum-1) );
	for( int i=0; i<partNum-1; i++ ) {
		matrix_new( &Cpiece[i], B->m, partSize );
	}
	matrix_new( &Cpiece[partNum-1], B->m, B->m-partSize*(partNum-1) );

	for( int i=0; i<partNum-1; i++ ) {
		extract_csr2csc<float>( &Bsub, B );
		copy_matrix_device( &Bsub );
		bhsparseSpgemm( &Cpiece[i], A, &Bsub );
		print_matrix( &Cpiece[i] );
		//print_array_device( "ColPtr", Cpiece[i].d_cscColPtr );
		//print_array_device( "RowInd", Cpiece[i].d_cscRowInd );
		//print_array_device( "Val", Cpiece[i].d_cscVal );
	}

	extract_csr2csc<float>( &Blast, B );
	copy_matrix_device( &Blast );
	bhsparseSpgemm( &Cpiece[partNum-1], A, &Blast );
}

int bhsparseSpgemm( d_matrix *C, d_matrix *A, d_matrix *B )
{
	int err = 0;
    bhsparse *bh_sparse = new bhsparse();

	bool *platforms = (bool*)malloc(NUM_PLATFORMS*sizeof(bool));
	for( int i=0; i<NUM_PLATFORMS; i++ ) platforms[i] = false;
	platforms[BHSPARSE_CUDA] = true;
    err = bh_sparse->initPlatform(platforms);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->initData(A->m, A->n, B->n,
                          A->nnz, (value_type *)A->h_cscVal, A->h_cscColPtr, A->h_cscRowInd,
                          B->nnz, (value_type *)B->h_cscVal, B->h_cscColPtr, B->h_cscRowInd,
                          C->h_cscColPtr);
    if(err != BHSPARSE_SUCCESS) return err;

    for (int i = 0; i < 3; i++)
    {
        err = bh_sparse->warmup();
        if(err != BHSPARSE_SUCCESS) return err;
    }

    err = bh_sparse->spgemm();
    if(err != BHSPARSE_SUCCESS) return err;

    // read back C
    C->nnz = bh_sparse->get_nnzC();
    C->h_cscRowInd = (int *)  malloc(C->nnz  * sizeof(int));
    C->h_cscVal    = (float *)  malloc(C->nnz  * sizeof(float));

    err = bh_sparse->get_C(C->h_cscRowInd, (value_type *)C->h_cscVal);
    if(err != BHSPARSE_SUCCESS) return err;

	return BHSPARSE_SUCCESS;
}

// Uses cuSPARSE SpGEMM
template< typename typeVal >
__global__ void populateRowIndVal( const int *d_cscColPtr, const int *d_cscRowInd, const typeVal *d_cscVal, int *d_dcscPartPtr, int *d_dcscColPtr_ind, int *d_dcscColPtr_off, int *d_dcscRowInd, typeVal *d_dcscVal, const int partNum, const int col_length, const int m, const int nnz )
{
	int start = threadIdx.x+blockIdx.x*blockDim.x;
	int stride = gridDim.x*blockDim.x;
	for( int idx=start; idx<m; idx+=stride )
	{
		int part = BinarySearchStart( d_dcscPartPtr, partNum+1, idx );

		int row_start = __ldg( d_cscColPtr+idx );
		int row_end = __ldg( d_cscColPtr+idx+1 );

		while( row_start < row_end )
		{
			int k = __ldg( d_cscRowInd+row_start );

			//int l = BinarySearchStart(
			// Need to have lock
		}
	}
}

__global__ void gather( const int *input_array, const int* indices, const int length, int *output_array )
{
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<length; idx+=blockDim.x*gridDim.x) {
        output_array[idx]=input_array[indices[idx]];
		printf("%d: %d %d\n", idx, indices[idx], output_array[idx]);
    }	
}

__global__ void shiftRightAdd( int *input_array, const int length )
{
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<length-1; idx+=blockDim.x*gridDim.x) {
        input_array[idx+1]=input_array[idx]+1;
    }	
}

__global__ void specialScatter( const int *input_array, const int partNum, int *output_array )
{
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<=partNum; idx+=blockDim.x*gridDim.x) {
        if( idx>0 ) output_array[input_array[idx]-1]=1;
    }	
}

// Uses MGPU SpMV
template<typename T>
void spmv( const T *d_inputVector, const int edge, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, edge, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

// Uses cuSPARSE SpGEMM
template<typename T>
void spgemm( d_matrix *C, d_matrix *A, d_matrix *B ) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);

    int baseC;
    int *nnzTotalDevHostPtr = &(C->nnz);
    cudaMalloc((void**) &(C->d_cscColPtr), (A->m+1)*sizeof(int));
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseStatus_t status = cusparseXcsrgemmNnz( handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              A->m, A->n, B->n,
                              descr, A->nnz,
                              A->d_cscColPtr, A->d_cscRowInd,
                              descr, B->nnz,
                              B->d_cscColPtr, B->d_cscRowInd,
                              descr,
                              C->d_cscColPtr, nnzTotalDevHostPtr );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("nnz count successful!\n");
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
    if( NULL != nnzTotalDevHostPtr )
        C->nnz = *nnzTotalDevHostPtr;
    else {
        cudaMemcpy( &(C->nnz), C->d_cscColPtr+C->m, sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( &baseC, C->d_cscColPtr, sizeof(int), cudaMemcpyDeviceToHost );
        C->nnz -= baseC;
    }
    printf("Output matrix: %d nnz\n", C->nnz);
    cudaMalloc((void**) &(C->d_cscRowInd), C->nnz*sizeof(int));
    cudaMalloc((void**) &(C->d_cscVal), C->nnz*sizeof(T));

    status                  = cusparseScsrgemm( handle,                   
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              A->m, A->n, B->n,
                              descr, A->nnz,
                              A->d_cscVal, A->d_cscColPtr, A->d_cscRowInd, 
                              descr, B->nnz,
                              B->d_cscVal, B->d_cscColPtr, B->d_cscRowInd,
                              descr,
                              C->d_cscVal, C->d_cscColPtr, C->d_cscRowInd );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("spgemm multiplication successful!\n");
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
