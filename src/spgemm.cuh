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

// Matrix multiplication (Host code)
void spgemmOuter( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {



}

// Matrix multiplication (Host code)
void spgemm( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {

	// Main goal 1: 
	//	-A in CSR -> A in DCSC
	//  -B in CSC -> B in DCSR
	csr_to_dcsc<float>( A, partSize, partNum, context, true );
	csr_to_dcsc<float>( A, partSize, partNum, context );

	csr_to_dcsc<float>( B, partSize, partNum, context, true );
	csr_to_dcsc<float>( B, partSize, partNum, context );

	// Main goal 2:
	//	-spgemmKernel
	//  -obtain C in CSR?

	// Some test arrays
	//int *d_output_triples;
	//cudaMalloc(&d_output_triples, A->nnz*sizeof(int));
	//float *d_output_total; 
	//cudaMalloc(&d_output_total, A->nnz*sizeof(float));

	// Copy length to host
	copy_part( A );
	copy_part( B );

	// Generate ColDiff
	int *d_dcscColDiffA, *d_dcscColDiffB;
    CUDA_SAFE_CALL(cudaMalloc(&d_dcscColDiffA, A->col_length*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dcscColDiffB, B->col_length*sizeof(int)));
    diff<<<BLOCKS,THREADS>>>( A->d_dcscColPtr_off, d_dcscColDiffA, A->col_length );
    diff<<<BLOCKS,THREADS>>>( B->d_dcscColPtr_off, d_dcscColDiffB, B->col_length );
	//print_array_device("DiffA", d_dcscColDiffA, A->col_length);
	//print_array_device("DiffB", d_dcscColDiffB, B->col_length);

	// Form array to see how long each intersection is
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
	printf("maxA:%d, maxB:%d\n", max_A, max_B);
	printf("avgA:%d, avgB:%d\n", A->col_length/partNum, B->col_length/partNum);
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
	//int numBlocks = MGPU_DIV_UP( max_A+max_B, NV );
	MGPU_MEM(int) countsDevice = context.Malloc<int>(numBlocks+1);
	CudaCheckError();

	// Allocate space for intersection indices
    int *d_intersectionA, *d_intersectionB, *d_interbalance, *d_scanbalance;
    CUDA_SAFE_CALL(cudaMalloc(&d_intersectionA, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_intersectionB, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_interbalance, A->col_length*partNum*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_scanbalance, A->col_length*partNum*sizeof(int)));

	// C->ColInd
	cudaMalloc(&(C->d_cscColInd), MAX_GLOBAL*sizeof(int));
	cudaMalloc(&(C->d_cscRowInd), MAX_GLOBAL*sizeof(int));
	cudaMalloc(&(C->d_cscVal), MAX_GLOBAL*sizeof(float));

	float elapsed = 0.0f;
    GpuTimer gpu_timer;
	gpu_timer.Start();

	for( int i=0; i<partNum; i++ ) {
		last_A = curr_A;
		curr_A = A->h_dcscPartPtr[i+1]-A->h_dcscPartPtr[i];
		curr_B = 0;
		for( int j=0; j<partNum; j++ ) {
			last_B = curr_B;
			curr_B = B->h_dcscPartPtr[j+1]-B->h_dcscPartPtr[j];
			//printf("i:%d, j:%d, LengthA:%d, LengthB:%d, Total:%d\n", i, j, curr_A, curr_B, total);
    		//total = mgpu::SetOpPairs<mgpu::MgpuSetOpIntersection, false>(A->d_dcscColPtr_ind+last_A,d_dcscColDiffA+last_A, curr_A, B->d_dcscColPtr_ind+last_B, d_dcscColDiffB+last_B, curr_B, d_intersectionA+h_inter[partNum*i+j], d_intersectionB+h_inter[partNum*i+j], d_interbalance+h_inter[partNum*i+j], &countsDevice, context);
    		//total = mgpu::SetOpPairs2<mgpu::MgpuSetOpIntersection, false>(A->d_dcscColPtr_ind+last_A,d_dcscColDiffA+last_A, curr_A, B->d_dcscColPtr_ind+last_B, d_dcscColDiffB+last_B, curr_B, d_intersectionA+h_inter[partNum*i+j], d_intersectionB+h_inter[partNum*i+j], d_interbalance+h_inter[partNum*i+j], &countsDevice, A, B, C, context);
			h_inter[i*partNum+j+1] = h_inter[i*partNum+j]+total;
		}
	}
	gpu_timer.Stop();
	elapsed += gpu_timer.ElapsedMillis();

	printf("intersection took: %f\n", elapsed);
	if( DEBUG ) { 
		print_array_device("intersectionA", d_intersectionA, h_inter[partNum*partNum]);
		print_array_device("intersectionB", d_intersectionB, h_inter[partNum*partNum]);
		print_array_device("interbalance", d_interbalance, h_inter[partNum*partNum]);
		print_array("intersection (first row scan)", h_inter, partNum+1);
		printf("intersection (total): %d\n", h_inter[partNum*partNum]); }

	// Important mallocs
    // Step 0: Preallocations for scratchpad memory
    /*int *d_flagArray, *d_tempArray, *d_lionArray, *d_sootArray;
	int *d_catmArray, *d_ouseArray, *d_rootArray, *d_beerArray;
	float *d_tempValA, *d_tempValB;
    CUDA_SAFE_CALL(cudaMalloc(&d_flagArray, A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_tempArray, 10*A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_lionArray, A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_sootArray, A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_catmArray, A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_ouseArray, 10*A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_rootArray, A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_beerArray, 70*A->nnz*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_tempValA, 70*A->nnz*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&d_tempValB, 10*A->nnz*sizeof(float)));*/

	// d_index
	int *d_index;
    CUDA_SAFE_CALL(cudaMalloc(&d_index, A->nnz*sizeof(int)));
    int *h_index = (int*)malloc(A->nnz*sizeof(int));
    for( int i=0; i<A->nnz;i++ ) h_index[i]=i;
    cudaMemcpy(d_index,h_index,A->nnz*sizeof(int),cudaMemcpyHostToDevice);

	// h_interbalance
	int *h_interbalance = (int*)malloc(h_inter[partNum*partNum]*sizeof(int));
	cudaMemcpy(h_interbalance, d_interbalance, h_inter[partNum*partNum]*sizeof(int), cudaMemcpyDeviceToHost);
	//print_array("interbalance", h_interbalance, h_inter[partNum*partNum]);

	float elapsed2 = 0.0f;
	GpuTimer gpu_timer2;
	gpu_timer2.Start();

	/*for( int i=0; i<1; i++ ) {
	//for( int i=0; i<partNum; i++ ) {

	int size = h_inter[(i+1)*partNum]-h_inter[i*partNum];
	int shift= h_inter[i*partNum];
	int lengthA = 0;
	int lengthB = 0;
	int length;

	CudaCheckError();
// Step 1: IntervalGather d_intersection: dcscColPtr_off => dcscColPtr_off (good)
	IntervalGather( size, d_intersectionA+shift, d_index, size, A->d_dcscColPtr_off, d_tempArray, context );
	IntervalGather( size, d_intersectionB+shift, d_index, size, B->d_dcscColPtr_off, d_flagArray, context );
	IntervalGather( size, d_intersectionA+shift, d_index, size, d_dcscColDiffA, d_lionArray, context );
	IntervalGather( size, d_intersectionB+shift, d_index, size, d_dcscColDiffB, d_sootArray, context );

// Step 1b: Must scan both good diffs - unnecessary since already have off good!
	mgpu::Scan<mgpu::MgpuScanTypeExc>( d_lionArray, size, 0, mgpu::plus<int>(), (int*)0, &(lengthA), d_rootArray, context );
	mgpu::Scan<mgpu::MgpuScanTypeExc>( d_sootArray, size, 0, mgpu::plus<int>(), (int*)0, &(lengthB), d_beerArray, context );
	mgpu::Scan<mgpu::MgpuScanTypeExc>( d_interbalance, h_inter[(i+1)*partNum], 0, mgpu::plus<int>(), (int*)0, &(length), d_scanbalance, context );
// But we need lengthA and lengthB!
// Can get away with just a reduce

// Step 1c: Must IntervalGather to form RowInd
	IntervalGather( lengthA, d_tempArray, d_rootArray, size, A->d_dcscRowInd, d_catmArray, context );
	IntervalGather( lengthA, d_tempArray, d_rootArray, size, A->d_dcscVal, d_tempValA, context );
	IntervalGather( lengthB, d_flagArray, d_beerArray, size, B->d_dcscRowInd, d_ouseArray, context );
	IntervalGather( lengthB, d_flagArray, d_beerArray, size, B->d_dcscVal, d_tempValB, context );
	CudaCheckError();
	printf("lengthA:%d, lengthB:%d, size:%d, shift:%d\n", lengthA, lengthB, size, shift);

// Step 2: Set-up for IntervalExpand A's RowInd
// Symmetric to here:
	if( length>A->nnz ) printf("Error: Length %d > %d\n", length, A->nnz);
	//IntervalExpand(lengthA, d_sootArray, d_dcscColDiffA, size, C->d_cscRowInd, context);
	IntervalExpand(lengthA, d_rootArray, d_sootArray, size, C->d_cscRowInd, context);
	mgpu::Scan<mgpu::MgpuScanTypeExc>( C->d_cscRowInd, lengthA, 0, mgpu::plus<int>(), (int*)0, &(length), d_tempArray, context );
	IntervalExpand(length, d_tempArray, d_catmArray, lengthA, C->d_cscRowInd, context);
	IntervalExpand(length, d_tempArray, d_tempValA, lengthA, C->d_cscVal, context);
	CudaCheckError();

// soot = ColDiffGoodB
// Step 3: Set-up for IntervalExpand B's RowInd
	IntervalExpand(lengthA, d_rootArray, d_sootArray, size, d_lionArray, context);
	IntervalExpand(lengthA, d_rootArray, d_beerArray, size, d_flagArray, context);
	mgpu::Scan<mgpu::MgpuScanTypeExc>( d_lionArray, lengthA, 0, mgpu::plus<int>(), (int*)0, &(length), d_beerArray, context);
	IntervalGather( length, d_flagArray, d_beerArray, lengthA, d_ouseArray, C->d_cscColInd, context );
	IntervalGather( length, d_flagArray, d_beerArray, lengthA, d_tempValB, d_tempValA, context );

// Step 4: need to sort pairs (C->d_cscColInd, d_tempValA)

// Step 5: ewiseMult into C->d_cscVal

	//h_interbalance[(i+1)*partNum];

	if( i==0 ) {
		print_array_device( "A RowInd temp", d_catmArray, lengthA);
		print_array_device( "B ColInd temp", d_ouseArray, lengthB);
		print_array_device( "C RowInd", C->d_cscRowInd, length);
		print_array_device( "C ColInd", C->d_cscColInd, length);
		print_array_device( "A RowVal", C->d_cscVal, length);
		print_array_device( "B ColVal", d_tempValA, length);
		print_array_device( "scanbalance", d_scanbalance, h_inter[(i+1)*partNum]);
		}
	}*/

	LaunchKernel<float>( C, A, B, d_intersectionA, d_intersectionB, d_interbalance, h_inter, max_AB, partSize, partNum, context );
	gpu_timer2.Stop();

	elapsed2 += gpu_timer2.ElapsedMillis();

	printf("outer product took: %f\n", elapsed2);
	//spgemmOuter( C, A, B, partSize, partNum, context );
}

/*// Matrix multiplication (Host code)
void spgemmInner( d_matrix *C, d_matrix *A, d_matrix *B, const int partSize, const int partNum, mgpu::CudaContext& context ) {

	// Some test arrays
	int *d_output_triples;
	cudaMalloc(&d_output_triples, A->nnz*sizeof(int));
	float *d_output_total; 
	cudaMalloc(&d_output_total, A->nnz*sizeof(float));

	int *zeroInt = (int*)malloc(A->nnz*sizeof(int));
	float *zeroFloat = (float*)malloc(A->nnz*sizeof(float));
	for( int i=0; i<A->nnz; i++ ) { zeroInt[i]=0; zeroFloat[i]=0.0; }
	cudaMemcpy(d_output_triples, zeroInt, A->nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_total, zeroFloat, A->nnz*sizeof(float), cudaMemcpyHostToDevice);

	// Make some useful const arrays
	int *h_nnzPartA = (int*)malloc(partNum*sizeof(int));
	int *h_nnzPartB = (int*)malloc(partNum*sizeof(int));
	int *h_numBlockA= (int*)malloc(partNum*sizeof(int));
	int *h_numBlockB= (int*)malloc(partNum*sizeof(int));
	h_nnzPartA[partNum-1] = A->h_cscColPtr[A->m]-A->h_cscColPtr[(partNum-1)*partSize];
	h_nnzPartB[partNum-1] = B->h_cscColPtr[B->m]-B->h_cscColPtr[(partNum-1)*partSize];
	h_numBlockA[partNum-1]= (h_nnzPartA[partNum-1]+SHARED-1)/SHARED;
	h_numBlockB[partNum-1]= (h_nnzPartB[partNum-1]+SHARED-1)/SHARED;
	int maxBlockA = h_numBlockA[partNum-1], maxBlockB = h_numBlockB[partNum-1];
	for( int i=0; i<partNum-2; i++ )
	{
		h_nnzPartA[i] = A->h_cscColPtr[(i+1)*partSize]-A->h_cscColPtr[i*partSize];
		h_nnzPartB[i] = B->h_cscColPtr[(i+1)*partSize]-B->h_cscColPtr[i*partSize];
		h_numBlockA[i]= (h_nnzPartA[i]+SHARED-1)/SHARED;
		h_numBlockB[i]= (h_nnzPartB[i]+SHARED-1)/SHARED;
		if( h_numBlockA[i]>maxBlockA ) maxBlockA = h_numBlockA[i];
		if( h_numBlockB[i]>maxBlockB ) maxBlockB = h_numBlockB[i];
	}
	int maxBlockAB = maxBlockA*maxBlockB;

	// Copy to device
	if( partNum > MAX_PART ){ printf("Error: Too many partitions!\n"); return;}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( nnzPartA, h_nnzPartA, partNum*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( nnzPartB, h_nnzPartB, partNum*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( numBlockA, h_numBlockA, partNum*sizeof(int) ));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( numBlockB, h_numBlockB, partNum*sizeof(int) ));
	int *d_nnzPartA, *d_nnzPartB, *d_numBlockA, *d_numBlockB;
	cudaMalloc( &d_nnzPartA, partNum*sizeof(int) );
	cudaMalloc( &d_nnzPartB, partNum*sizeof(int) );
	cudaMalloc( &d_numBlockA, partNum*sizeof(int) );
	cudaMalloc( &d_numBlockB, partNum*sizeof(int) );
	CUDA_SAFE_CALL(cudaMemcpy( d_nnzPartA, h_nnzPartA, partNum*sizeof(int), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy( d_nnzPartB, h_nnzPartB, partNum*sizeof(int), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy( d_numBlockA, h_numBlockA, partNum*sizeof(int), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy( d_numBlockB, h_numBlockB, partNum*sizeof(int), cudaMemcpyHostToDevice ));

	printf("maxA:%d, maxB:%d, first block threads:%d, blocks:%d\n", maxBlockA, maxBlockB, h_numBlockA[0]*h_numBlockB[0], h_numBlockA[0]*h_numBlockB[0]*SHARED);
	print_array("number of nnz in A", h_numBlockA, 5);
	print_array("number of nnz in B", h_numBlockB, 5);
	print_array_device("A ColPtr", A->d_cscColPtr, A->m+1);
	print_array_device("A RowInd", A->d_cscRowInd, A->nnz);
	print_array_device("B ColPtr", B->d_cscColPtr, B->m+1);
	print_array_device("B RowInd", B->d_cscRowInd, B->nnz);
	long tc_count = LaunchKernel<float>( C, A, B, d_output_triples, d_output_total, partSize, partNum, maxBlockA, maxBlockB, maxBlockAB, context );
	print_array_device(A->d_cscColPtr, A->m+1);
	print_array_device(A->d_cscRowInd, A->nnz);
	print_array_device(B->d_cscColPtr, B->m+1);
	print_array_device(B->d_cscRowInd, B->nnz);
}*/

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
