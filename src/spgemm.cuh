// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include <cudpp.h>
#include "mXv.cuh"
#include "scratch.hpp"
#include "matrix.hpp"
#include "spgemmKernel.cuh"
#include "common.hpp"

#define NTHREADS 512
#define MAX_SHARED 49152
#define MAX_GLOBAL 270000000

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


// Matrix multiplication (Host code)
void spgemmOuter( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {

}

__global__ void updateInter( int *d_intersectionA, const int numThread, const int offset ) {
    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx<=numThread; idx+=blockDim.x*gridDim.x)
		d_intersectionA[idx] += offset;
}

__global__ void add( int *d_moveCount, const int length ) {
	int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if( gid<length && d_moveCount[gid]!=0 ) {
		d_moveCount[gid]+=d_moveCount[gid+length];
	}
}

void compareCudpp( const int *d_lengthA, int *d_scanbalance, int *h_inter, int *d_moveCount, const int numInter, const int partNum, mgpu::CudaContext &context ) {

	unsigned *d_flag;
	CUDA_SAFE_CALL(cudaMalloc( &d_flag, numInter*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMemset( d_flag, (unsigned) 0, numInter*sizeof(unsigned) ));

	unsigned one = 1;

	for( int i=0; i<partNum*partNum-1; i++ ) {
		//printf("i:%d, h_inter:%d\n", i, h_inter[i+1]);
		CUDA_SAFE_CALL(cudaMemcpy( d_flag+h_inter[i+1], &one, sizeof(unsigned),
			cudaMemcpyHostToDevice ));
	}

	// CUDPP code
	CUDPPHandle theCudpp;
	cudppCreate(&theCudpp);

 	CUDPPConfiguration config;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_INT;
	config.algorithm = CUDPP_SEGMENTED_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	CUDPPHandle scanplan = 0;

	CUDPPResult res = cudppPlan( theCudpp, &scanplan, config, numInter, 1, 0 );

	if (CUDPP_SUCCESS != res) {
		printf("Error creating CUDPPPlan\n");
		exit(-1);
	}

	// Run the scan
	res = cudppSegmentedScan(scanplan, (void*)d_scanbalance, (void*)d_lengthA, 
		d_flag, numInter);
	if (CUDPP_SUCCESS != res)
	{
		printf("Error in cudppScan()\n");
		exit(-1);
	}

	int *h_inter2 = (int*) malloc( (partNum*partNum+1)*sizeof(int) );
	memcpy( h_inter2, h_inter, (partNum*partNum+1)*sizeof(int) );
	if(DEBUG_SPGEMM) print_array( "h_inter", h_inter2, partNum*partNum+1 );
	for( int i=0; i<partNum*partNum; i++ ) {
		if( h_inter2[i]==h_inter2[i+1] ) h_inter2[i] = 0;
		else h_inter2[i] = h_inter2[i+1]-1;
	}
	if(DEBUG_SPGEMM) print_array( "h_inter2", h_inter2, partNum*partNum+1 );

	int *d_inter;
	CUDA_SAFE_CALL(cudaMalloc( &d_inter, partNum*partNum*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMemcpy( d_inter, h_inter2, partNum*partNum*sizeof(int), 
		cudaMemcpyHostToDevice ));

	IntervalGather( partNum*partNum, d_inter, mgpu::counting_iterator<int>(0), 
		partNum*partNum, d_scanbalance, d_moveCount, context );
	IntervalGather( partNum*partNum, d_inter, mgpu::counting_iterator<int>(0),
		partNum*partNum, d_lengthA, d_moveCount+partNum*partNum, context );
	const int NBLOCKS = (partNum*partNum+NTHREADS-1)/NTHREADS;
	add<<<NBLOCKS,NTHREADS>>>( d_moveCount, partNum*partNum );

	if(DEBUG_SPGEMM) print_array_device( "d_inter", d_inter, partNum*partNum );
	if(DEBUG_SPGEMM) print_array_device( "d_moveCount", d_moveCount, partNum*partNum );
	if(DEBUG_SPGEMM) print_array_device( "scan", d_scanbalance, numInter+1 );
}


// Matrix multiplication (Host code)
void spgemm( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {

	// Stage 1: 
	//	-A in CSR -> A in DCSC
	//  -B in CSC -> B in DCSR
	printf("==Stage 1==\n");
	csr_to_dcsc<float>( A, partSize, partNum, context, true );
	csr_to_dcsc<float>( A, partSize, partNum, context );

	csr_to_dcsc<float>( B, partSize, partNum, context, true );
	csr_to_dcsc<float>( B, partSize, partNum, context );

	// Stage 2:
	// -form array to see how long each intersection is
	// -compute intersection
	float elapsed = 0.0f;
    GpuTimer gpu_timer;
	gpu_timer.Start();

	// Copy length to host
	copy_part( A );
	copy_part( B );

	// Generate ColDiff
	int *d_dcscColDiffA, *d_dcscColDiffB;
    CUDA_SAFE_CALL(cudaMalloc(&d_dcscColDiffA, A->col_length*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dcscColDiffB, B->col_length*sizeof(int)));
    diff<<<BLOCKS,THREADS>>>( A->d_dcscColPtr_off, d_dcscColDiffA, A->col_length );
    diff<<<BLOCKS,THREADS>>>( B->d_dcscColPtr_off, d_dcscColDiffB, B->col_length );

	int *h_inter = (int*)malloc((partNum*partNum+1)*sizeof(int));

	h_inter[0] = 0;
	int total=0;
	int curr_A=0; int curr_B=0;
	int last_A; int last_B;
	int max_A=0; int max_B=0;
	for( int i=0; i<partNum; i++ ) { 
		last_A = A->h_dcscPartPtr[i+1]-A->h_dcscPartPtr[i];
		last_B = B->h_dcscPartPtr[i+1]-B->h_dcscPartPtr[i];
		//printf("LengthA:%d, LengthB:%d\n", last_A, last_B);
		if( last_A > max_A ) max_A = last_A;
		if( last_B > max_B ) max_B = last_B;
	}
	if(DEBUG_KERNEL) printf("maxA:%d, maxB:%d\n", max_A, max_B);
	if(DEBUG_KERNEL) printf("avgA:%d, avgB:%d\n", A->col_length/partNum, B->col_length/partNum);
	int max_AB = min(max_A,max_B);

	// Allocate maximum number of blocks
    typedef mgpu::LaunchBoxVT<
        128, 23, 0,
        128, 11, 0,
        128, 11, 0
    > Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    const int NV = launch.x * launch.y;
	int numBlocks = MGPU_DIV_UP( max_A+B->col_length, NV );
	MGPU_MEM(int) countsDevice = context.Malloc<int>(numBlocks+1);
	CudaCheckError();

	// Allocate space for intersection indices
    //   d_intersectionA- This is a list of indices that are in both A and B
	//   d_interbalance - This is how much work is done in each square
    int *d_intersectionA, *d_intersectionB, *d_interbalance, *d_scanbalance;
    CUDA_SAFE_CALL(cudaMalloc(&d_intersectionA, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_intersectionB, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_interbalance, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_scanbalance, A->col_length*partNum*sizeof(int)));

	// Allocate C->d_cscColInd
	cudaMalloc(&(C->d_cscColInd), MAX_GLOBAL*sizeof(int));
	cudaMalloc(&(C->d_cscRowInd), MAX_GLOBAL*sizeof(int));
	cudaMalloc(&(C->d_cscVal), MAX_GLOBAL*sizeof(float));

	for( int i=0; i<partNum; i++ ) {
		last_A = curr_A;
		curr_A = A->h_dcscPartPtr[i+1]-A->h_dcscPartPtr[i];
		curr_B = 0;
		for( int j=0; j<partNum; j++ ) {
			last_B = curr_B;
			curr_B = B->h_dcscPartPtr[j+1]-B->h_dcscPartPtr[j];
			//printf("i:%d, j:%d, LengthA:%d, LengthB:%d, Total:%d\n", i, j, curr_A, 
			//	curr_B, total);
    		total = mgpu::SetOpPairs<mgpu::MgpuSetOpIntersection, false>(
				A->d_dcscColPtr_ind+last_A,d_dcscColDiffA+last_A, curr_A, 
				B->d_dcscColPtr_ind+last_B, d_dcscColDiffB+last_B, curr_B, 
				d_intersectionA+h_inter[partNum*i+j], d_intersectionB+
				h_inter[partNum*i+j], d_interbalance+h_inter[partNum*i+j], 
				&countsDevice, context);

			h_inter[i*partNum+j+1] = h_inter[i*partNum+j]+total;
			updateInter<<<BLOCKS,NTHREADS>>>( d_intersectionB+h_inter[partNum*i+j], 
				total, B->h_dcscPartPtr[j] );
		}
		updateInter<<<BLOCKS, NTHREADS>>>( d_intersectionA+h_inter[partNum*i], 
			h_inter[partNum*(i+1)]-h_inter[partNum*(i)], A->h_dcscPartPtr[i] );
	}
	gpu_timer.Stop();
	elapsed += gpu_timer.ElapsedMillis();

	printf("==Stage 2==\n");
	printf("Intersection: %f\n", elapsed);
	if( DEBUG_SPGEMM ) { 
		print_array_device("intersectionA", d_intersectionA, h_inter[partNum*partNum]);
		print_array_device("intersectionB", d_intersectionB, h_inter[partNum*partNum]);
		print_array_device("interbalance", d_interbalance, h_inter[partNum*partNum]);
		printf("intersection (total): %d\n", h_inter[partNum*partNum]); }

	// Stage 3:
	//
	// Important mallocs
    // Step 0: Preallocations for scratchpad memory
	//
	// d_index - 0, 1, 2, 3, 4, ...
    // d_ones  - 1, 1, 1, 1, 1, ...
	float elapsed3 = 0.0f;
	GpuTimer gpu_timer3;
	gpu_timer3.Start();

	int *d_index;
    CUDA_SAFE_CALL(cudaMalloc(&d_index, A->nnz*sizeof(int)));
    int *h_index = (int*)malloc(A->nnz*sizeof(int));
    for( int i=0; i<A->nnz;i++ ) h_index[i]=i;
    CUDA_SAFE_CALL(cudaMemcpy(d_index,h_index,A->nnz*sizeof(int),
		cudaMemcpyHostToDevice));

	int *d_ones, *h_ones;
	h_ones = (int*) malloc( A->nnz*sizeof(int) );
	for( int i=0; i<A->nnz; i++ ) h_ones[i] = 1;
	CUDA_SAFE_CALL(cudaMalloc(&d_ones, A->nnz*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy( d_ones, h_ones, A->nnz*sizeof(int), 
		cudaMemcpyHostToDevice )); 

	// TODO: compute:
	//	1) numInterA - this is number of intersections in A
	//  2) d_colDiffA- this is the length of each column in A
    //  3) d_lengthA - this is the 
	//  4) fix d_intersectionA being local (need to add partPtr[i] to it)

	int numInter = h_inter[partNum*partNum];

	int *d_colDiffA;
	CUDA_SAFE_CALL(cudaMalloc( &d_colDiffA, A->col_length*sizeof(int) ));
	int NBLOCKS = (A->col_length+NTHREADS-1)/NTHREADS;
	diff<<<NBLOCKS,NTHREADS>>>( A->d_dcscColPtr_off, d_colDiffA, A->col_length );
	if(DEBUG_SPGEMM) print_array_device("colDiffA", d_colDiffA, A->col_length);

	int *d_lengthA, *d_offA;
	CUDA_SAFE_CALL(cudaMalloc( &d_lengthA, numInter*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMalloc( &d_offA, numInter*sizeof(int) ));

	// Step 1: Form d_offA, d_lengthA, d_offB, d_lengthB
    IntervalGather( numInter, d_intersectionA, mgpu::counting_iterator<int>(0), 
		numInter, d_colDiffA, d_lengthA, context );
	IntervalGather( numInter, d_intersectionA, mgpu::counting_iterator<int>(0), 
		numInter, A->d_dcscColPtr_off, d_offA, context );
	if(DEBUG_SPGEMM) print_array_device("lengthA", d_lengthA, numInter);
	if(DEBUG_SPGEMM) print_array_device("offA", d_offA, numInter);

	int *d_colDiffB;
	CUDA_SAFE_CALL(cudaMalloc( &d_colDiffB, B->col_length*sizeof(int) ));
	NBLOCKS = (B->col_length+NTHREADS-1)/NTHREADS;
	diff<<<NBLOCKS,NTHREADS>>>( B->d_dcscColPtr_off, d_colDiffB, B->col_length );
	if(DEBUG_SPGEMM) print_array_device("colDiffB", d_colDiffB, B->col_length);

	int *d_lengthB, *d_offB;
	CUDA_SAFE_CALL(cudaMalloc( &d_lengthB, numInter*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMalloc( &d_offB, numInter*sizeof(int) ));

    IntervalGather( numInter, d_intersectionB, mgpu::counting_iterator<int>(0), 
		numInter, d_colDiffB, d_lengthB, context );
	IntervalGather( numInter, d_intersectionB, mgpu::counting_iterator<int>(0), 
		numInter, B->d_dcscColPtr_off, d_offB, context );
	if(DEBUG_SPGEMM) print_array_device("lengthB", d_lengthB, numInter);
	if(DEBUG_SPGEMM) print_array_device("offB", d_offB, numInter);

	// Step 2:
	//
	// input array: d_lengthA
	// input scan: d_inter
	// output array: d_scanbalance
	// output moveCount: d_moveCount
	// numInter: h_inter[partNum*partNum]
	int *d_moveCount, *h_moveCount, moveCount;
	CUDA_SAFE_CALL(cudaMalloc( &d_moveCount, 2*partNum*partNum*sizeof(int) ));
	h_moveCount = (int*) malloc (partNum*partNum*sizeof(int) );

	compareCudpp( d_lengthA, d_scanbalance, h_inter, d_moveCount, numInter, 
		partNum, context);
	gpu_timer3.Stop();
	elapsed3 += gpu_timer3.ElapsedMillis();
	printf("==Stage 3==\n");
	printf("Segmented Scan: %f\n", elapsed3);
	CUDA_SAFE_CALL(cudaMemcpy( h_moveCount, d_moveCount, partNum*partNum*
		sizeof(int), cudaMemcpyDeviceToHost ));
    mgpu::Reduce( d_moveCount, partNum*partNum, (int)0, mgpu::plus<int>(), (int*)0, 
		&moveCount, context );
	CudaCheckError();

	float elapsed2 = 0.0f;
	GpuTimer gpu_timer2;
	gpu_timer2.Start();

	LaunchKernel<float>( C, A, B, d_offA, d_lengthA, d_offB, d_lengthB, 
		d_interbalance, d_scanbalance, h_inter, d_moveCount, h_moveCount, 
		partSize, partNum, moveCount, context );
	gpu_timer2.Stop();

	elapsed2 += gpu_timer2.ElapsedMillis();

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
