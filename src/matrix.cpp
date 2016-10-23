#include <iomanip>
#include <cub/cub.cuh>
#include <moderngpu.cuh>

#include "matrix.hpp"

#define DEBUG 1

void matrix_new( d_matrix *A, int m, int n )
{
	A->m = m;
	A->n = n;

	// Host alloc
    A->h_cscColPtr = (int*)malloc((A->m+1)*sizeof(int));

	// RowInd and Val will be allocated in buildMatrix rather than here
	// since nnz may be unknown
    A->h_cscRowInd = NULL;
	A->h_cscVal = NULL;
	A->d_cscRowInd = NULL;
	A->d_cscVal = NULL;
	A->d_dcscPartPtr = NULL;
	A->d_dcscRowInd = NULL;
	A->d_dcscVal = NULL;

	// Device alloc
    cudaMalloc(&(A->d_cscColPtr), (A->m+1)*sizeof(int));
	cudaMalloc(&(A->d_dcscColPtr_ind), A->DCSC*min(A->m,A->n)*sizeof(int));
	cudaMalloc(&(A->d_dcscColPtr_off), A->DCSC*min(A->m,A->n)*sizeof(int));
}

// This function converts function from COO to CSC/CSR representation
//	 Usage: buildMatrix<typeVal( A, edge, h_cooColIndA, h_cooRowIndA, h_cooValA )
//			=> CSC
//			buildMatrix<typeVal( A, edge, h_cooRowIndA, h_cooColIndA, h_cooValA )
//			=> CSR
// TODO: -add support for int64
//
// @tparam[in] <typeVal>   Models value
//
// @param[in] A
// @param[in] numEdge 
// @param[in] h_cooRowInd
// @param[in] h_cooColInd
// @param[in] h_cooVal
// @param[out] A

void matrix_delete( d_matrix *A )
{
	cudaFree( A->d_cscColPtr );
	if( A->d_cscRowInd != NULL ) cudaFree( A->d_cscRowInd );
	if( A->d_cscVal != NULL ) cudaFree( A->d_cscVal );
}

template<typename typeVal>
void buildMatrix( d_matrix *A,
                    int numEdge,
                    int *h_cooRowInd,     // I
                    int *h_cooColInd,     // J
                    typeVal *h_cooVal ) {

	A->nnz = numEdge;
	
	// Host malloc
    A->h_cscRowInd = (int*)malloc(A->nnz*sizeof(int));
    A->h_cscVal = (typeVal*)malloc(A->nnz*sizeof(typeVal));	

	// Device malloc
    cudaMalloc(&(A->d_cscVal), A->nnz*sizeof(typeVal));
    cudaMalloc(&(A->d_cscRowInd), A->nnz*sizeof(int));

	// DCSC malloc
    cudaMalloc(&(A->d_dcscVal), A->nnz*sizeof(typeVal));
    cudaMalloc(&(A->d_dcscRowInd), A->nnz*sizeof(int));

	// Convert to CSC/CSR
    int temp;
    int row;
    int dest;
    int cumsum = 0;

    for( int i=0; i<=A->m; i++ )
      A->h_cscColPtr[i] = 0;               // Set all rowPtr to 0
    for( int i=0; i<A->nnz; i++ )
      A->h_cscColPtr[h_cooRowInd[i]]++;                   // Go through all elements to see how many fall into each column
    for( int i=0; i<A->m; i++ ) {                  // Cumulative sum to obtain column pointer array
      temp = A->h_cscColPtr[i];
      A->h_cscColPtr[i] = cumsum;
      cumsum += temp;
    }
    A->h_cscColPtr[A->m] = A->nnz;

    for( int i=0; i<A->nnz; i++ ) {
      row = h_cooRowInd[i];                         // Store every row index in memory location specified by colptr
      dest = A->h_cscColPtr[row];
      A->h_cscRowInd[dest] = h_cooColInd[i];              // Store row index
      A->h_cscVal[dest] = h_cooVal[i];                 // Store value
      A->h_cscColPtr[row]++;                      // Shift destination to right by one
    }
    cumsum = 0;
    for( int i=0; i<=A->m; i++ ) {                 // Undo damage done by moving destination
      temp = A->h_cscColPtr[i];
      A->h_cscColPtr[i] = cumsum;
      cumsum = temp;
	}

	// Device memcpy
    cudaMemcpy(A->d_cscVal, A->h_cscVal, A->nnz*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(A->d_cscRowInd, A->h_cscRowInd, A->nnz*sizeof(int),cudaMemcpyHostToDevice);	
    cudaMemcpy(A->d_cscColPtr, A->h_cscColPtr, (A->m+1)*sizeof(int),cudaMemcpyHostToDevice);
}

// This function takes a matrix A that's already been buildMatrix'd and performs
// deep copy to B
// 
// TODO: -add dimension mismatch checks
//		 -convert to C11 _generic semantics
//
// @param[in]  A
// @param[out] B
void matrix_copy( d_matrix *B, d_matrix *A )
{
	B->nnz = A->nnz;

	// Host alloc
    B->h_cscColPtr = (int*)malloc((B->m+1)*sizeof(int));
    B->h_cscRowInd = (int*)malloc(B->nnz*sizeof(int));
    B->h_cscVal = (float*)malloc(B->nnz*sizeof(float));	

	// Device alloc
    cudaMalloc(&(B->d_cscColPtr), (B->m+1)*sizeof(int));
    cudaMalloc(&(B->d_cscVal), B->nnz*sizeof(float));
    cudaMalloc(&(B->d_cscRowInd), B->nnz*sizeof(int));

	// Host memcpy
    memcpy( B->h_cscColPtr, A->h_cscColPtr, (B->m+1)*sizeof(int));
    memcpy( B->h_cscRowInd, A->h_cscRowInd, B->nnz*sizeof(int));
    memcpy( B->h_cscVal, A->h_cscVal, B->nnz*sizeof(float));

	// Device memcpy
    cudaMemcpy(B->d_cscColPtr, A->h_cscColPtr, (B->m+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(B->d_cscVal, A->h_cscVal, B->nnz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B->d_cscRowInd, A->h_cscRowInd, B->nnz*sizeof(int),cudaMemcpyHostToDevice);
}

void print_matrix( d_matrix *A, bool val ) {
    std::cout << "Matrix:\n";
	int num_row = A->m>20 ? 20 : A->m;
	int num_col = A->n>20 ? 20 : A->n; 
    for( int i=0; i<num_row; i++ ) {
        int count = A->h_cscColPtr[i];
        for( int j=0; j<num_col; j++ ) {
            if( count>=A->h_cscColPtr[i+1] || A->h_cscRowInd[count] != j )
                std::cout << "0 ";
            else {
				if( val )
                	std::cout << std::setprecision(2) << A->h_cscVal[count] << " ";
				else
					std::cout << "x ";
                count++;
            }
        }
        std::cout << std::endl;
    }
}

void copy_matrix_device( d_matrix *A ) {

	// If buildMatrix not run, then need host alloc
	// Both pointers set to NULL in matrix_new 
	if( A->h_cscRowInd == NULL && A->h_cscVal == NULL )
	{
		//std::cout << "Allocating memory for print.\n";
    	A->h_cscRowInd = (int*)malloc(A->nnz*sizeof(int));
    	A->h_cscVal = (float*)malloc(A->nnz*sizeof(float));	
	}

	// Copy from device
    cudaMemcpy(A->h_cscVal, A->d_cscVal, A->nnz*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(A->h_cscRowInd, A->d_cscRowInd, A->nnz*sizeof(int),cudaMemcpyDeviceToHost);	
    cudaMemcpy(A->h_cscColPtr, A->d_cscColPtr, (A->m+1)*sizeof(int),cudaMemcpyDeviceToHost);
}

void print_matrix_device( d_matrix *A, bool val ) {

	// If buildMatrix not run, then need host alloc
	// Both pointers set to NULL in matrix_new 
	if( A->h_cscRowInd == NULL && A->h_cscVal == NULL )
	{
		//std::cout << "Allocating memory for print.\n";
    	A->h_cscRowInd = (int*)malloc(A->nnz*sizeof(int));
    	A->h_cscVal = (float*)malloc(A->nnz*sizeof(float));	
	}

	// Copy from device
    cudaMemcpy(A->h_cscVal, A->d_cscVal, A->nnz*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(A->h_cscRowInd, A->d_cscRowInd, A->nnz*sizeof(int),cudaMemcpyDeviceToHost);	
    cudaMemcpy(A->h_cscColPtr, A->d_cscColPtr, (A->m+1)*sizeof(int),cudaMemcpyDeviceToHost);
	
	print_matrix( A, val );
}

// extract( Asub, A ) - extract subgraph B of A
template<typename typeVal>
void extract( d_matrix *B, const d_matrix *A )
{
    // Allocate memory for B->nnz
    B->nnz = A->h_cscColPtr[B->m];
	//std::cout << B->m << " " << B->n << " " << B->nnz << " \n";

	if( B->h_cscRowInd == NULL && B->h_cscVal == NULL )
	{
		B->h_cscRowInd = (int*)malloc(B->nnz*sizeof(int));
    	B->h_cscVal = (typeVal*)malloc(B->nnz*sizeof(typeVal));

    	cudaMalloc( &(B->d_cscRowInd), B->nnz*sizeof(int) );
    	cudaMalloc( &(B->d_cscVal), B->nnz*sizeof(typeVal) );
	}

    cudaMemcpy(B->d_cscColPtr, A->d_cscColPtr, (B->m+1)*sizeof(int),cudaMemcpyDeviceToDevice);
    cudaMemcpy(B->d_cscVal, A->d_cscVal, B->nnz*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy(B->d_cscRowInd, A->d_cscRowInd, B->nnz*sizeof(int),cudaMemcpyDeviceToDevice);

}

// Special case of csr2csc that fuses:
// 1. extract( Asub, A ) - extract subgraph B of A
// 2. csr2csc( B, Asub ) - convert subgraph to CSC
template<typename typeVal>
void extract_csr2csc( d_matrix *B, const d_matrix *A )
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Allocate memory for B->nnz
    B->nnz = A->h_cscColPtr[B->n];
	//std::cout << B->m << " " << B->n << " " << B->nnz << " \n";

	B->h_cscRowInd = (int*)malloc(B->nnz*sizeof(int));
    B->h_cscVal = (typeVal*)malloc(B->nnz*sizeof(typeVal));

    cudaMalloc( &(B->d_cscRowInd), B->nnz*sizeof(int) );
    cudaMalloc( &(B->d_cscVal), B->nnz*sizeof(typeVal) );

	//print_array_device(A->d_cscColPtr,40);
	//print_array_device(A->d_cscRowInd,40);
	//print_array_device(A->d_cscVal,40);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsr2csc(handle, B->n, B->m, B->nnz, A->d_cscVal, A->d_cscColPtr, A->d_cscRowInd, B->d_cscVal, B->d_cscRowInd, B->d_cscColPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("csr2csc conversion successful!\n");
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
	}

	//printf("Matrix B:\n");
	//print_array_device(B->d_cscColPtr,40);
	//print_array_device(B->d_cscRowInd,40);
	//print_array_device(B->d_cscVal,40);

    // Important: destroy handle
    cusparseDestroy(handle);
}

// Wrapper for device code
template <typename typeVal>
void csr_to_dcsc( d_matrix *A, const int partSize, const int partNum, mgpu::CudaContext& context, bool alloc )
{
	if( alloc )
	{
		A->part = partNum;

		A->h_dcscPartPtr = (int*)malloc((partNum+1)*sizeof(int));
		cudaMalloc(&(A->d_dcscPartPtr), (partNum+1)*sizeof(int));
	} else {
	// Step 0: Preallocations for scratchpad memory
	int *d_flagArray, *d_tempArray, *d_index;
	CUDA_SAFE_CALL(cudaMalloc(&d_flagArray, A->nnz*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&d_tempArray, A->nnz*sizeof(int)));

		// d_index
		CUDA_SAFE_CALL(cudaMalloc(&d_index, A->nnz*sizeof(int)));
		int *h_index = (int*)malloc(A->nnz*sizeof(int));
		for( int i=0; i<A->nnz;i++ ) h_index[i]=i;
		cudaMemcpy(d_index,h_index,A->nnz*sizeof(int),cudaMemcpyHostToDevice);

		// d_cscColDiff
		/*int *d_cscColDiff;
		CUDA_SAFE_CALL(cudaMalloc(&d_cscColDiff, A->m*sizeof(int)));
		diff<<<BLOCKS,THREADS>>>( A->d_cscColPtr, d_cscColDiff, A->m );*/

	float elapsed1 = 0.0f;
	float elapsed2 = 0.0f;
	float elapsed3 = 0.0f;
	float elapsed4 = 0.0f;
	float elapsed5 = 0.0f;
	GpuTimer gpu_timer1;
	GpuTimer gpu_timer2;
	GpuTimer gpu_timer3;
	GpuTimer gpu_timer4;
	GpuTimer gpu_timer5;
	gpu_timer1.Start();
	// Step 1: Segmented Sort RowInd of CSC 
	// Set up parameters
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;

		// CUDA mallocs
		int *h_offsets = (int*)malloc((partNum+1)*sizeof(int));
		for( int i=0; i<partNum; i++ ) h_offsets[i]=A->h_cscColPtr[i*partSize];
		h_offsets[partNum] = A->nnz;
		int *d_offsets;
		cudaMalloc(&d_offsets, (partNum+1)*sizeof(int));
		cudaMemcpy(d_offsets,h_offsets,(partNum+1)*sizeof(int),cudaMemcpyHostToDevice);

		// Upper bit limit radix sort: Use log(A->n)
		int end_bit = log2( *(uint32_t*)&(A->n) )+1;

		// Determine temporary device storage requirements
		if( DEBUG ) print_array_device("last 40 of output", A->d_dcscRowInd, A->nnz);
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, A->d_cscRowInd, A->d_dcscRowInd, A->nnz, partNum, d_offsets, d_offsets + 1, 0, end_bit);
		
		// Allocate temporary storage
		CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		
		// Run sorting operation
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, A->d_cscRowInd, A->d_dcscRowInd, A->nnz, partNum, d_offsets, d_offsets + 1, 0, end_bit);
		if( DEBUG ) print_array_device("last 40 of output", A->d_cscRowInd, A->nnz );
		//CudaCheckError();

		// Check results
		CUDA_SAFE_CALL(cudaFree(d_temp_storage));
		d_temp_storage = NULL;
	gpu_timer1.Stop();
	gpu_timer2.Start();

	// Step 2: Use scan to get d_dcscColPtr_off

		// Obtain d_dcscColPtr_off
		lookRight<<<BLOCKS,THREADS>>>( A->d_dcscRowInd, A->nnz, d_flagArray );

		// Need to do some fix-ups:
		// 1) Set all h_offset[i]-1 to 1
		if( DEBUG ) print_array_device("flag_array before", d_flagArray, A->nnz);
		specialScatter<<<1,THREADS>>>( d_offsets, partNum, d_flagArray );
		if( DEBUG ) print_array_device("flag_array after", d_flagArray, A->nnz);
		if( DEBUG ) CudaCheckError();
		mgpu::Scan<mgpu::MgpuScanTypeExc>( d_flagArray, A->nnz, 0, mgpu::plus<int>(), (int*)0, &(A->col_length), d_tempArray, context );
		if( A->col_length > A->DCSC*min(A->m,A->n) ) { printf("Error: array %d too long for %d\n", A->col_length, A->DCSC*min(A->m,A->n)); return; }
	 	streamCompact<<<BLOCKS,THREADS>>>( d_flagArray, d_tempArray, A->d_dcscColPtr_off, A->nnz );

	gpu_timer2.Stop();
	gpu_timer3.Start();

	// Step 3: Use d_dcscColPtr_off and segmented sort output to get d_dcscColPtr_ind
		IntervalGather( A->col_length, A->d_dcscColPtr_off, d_index, A->col_length, A->d_dcscRowInd, A->d_dcscColPtr_ind, context );
		shiftRight<<<BLOCKS,THREADS>>>( A->d_dcscColPtr_off, A->col_length );
		cudaMemset( A->d_dcscColPtr_off, 0, sizeof(int));
		if( DEBUG ) CudaCheckError();

	gpu_timer3.Stop();
		if( DEBUG ) print_array_device("offset array", d_offsets, partNum+1);
		if( DEBUG ) print_array_device("dcscPartPtr", A->d_dcscPartPtr, partNum+1);
	gpu_timer4.Start();

	// Step 4: Segmented Reduce to find d_dcscPartPtr
		IntervalGather( partNum, d_offsets, d_index, partNum, d_tempArray, A->d_dcscPartPtr, context );
		if( DEBUG ) CudaCheckError();
		cudaMemcpy( A->d_dcscPartPtr+partNum, &(A->col_length), sizeof(int), cudaMemcpyHostToDevice );

	gpu_timer4.Stop();
	gpu_timer5.Start();

	// Step 5: Populate RowInd and Val

		if( DEBUG ) print_array_device("Index", d_index, A->m);
		if( DEBUG ) print_array_device("ColPtr", A->d_cscColPtr, A->m);
		if( DEBUG ) print_array_device("RowInd", A->d_cscRowInd, A->nnz);
		if( DEBUG ) CudaCheckError();

		// Generate cscColInd (d_tempArray)
		IntervalExpand( A->nnz, A->d_cscColPtr, d_index, A->m, d_tempArray, context );
		if( DEBUG ) CudaCheckError();

		// Determine temporary device storage requirements
		cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, A->d_cscRowInd, A->d_dcscRowInd, d_index, d_flagArray, A->nnz, partNum, d_offsets, d_offsets + 1, 0, end_bit);
		if( DEBUG ) CudaCheckError();
		
		// Allocate temporary storage
		CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		if( DEBUG ) printf("%d storage bytes\n", temp_storage_bytes);
		if( DEBUG ) CudaCheckError();
		
		// Run sorting operation
		cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, A->d_cscRowInd, A->d_dcscRowInd, d_index, d_flagArray, A->nnz, partNum, d_offsets, d_offsets + 1, 0, end_bit);
		if( DEBUG ) CudaCheckError();

		if( DEBUG ) print_array_device("New Map", d_flagArray, A->nnz);
		if( DEBUG ) print_array_device("Col Ind", d_tempArray, A->nnz);
		if( DEBUG ) CudaCheckError();

		// Gather A->cscColInd and A->cscVal into indices specified by d_flagArray
		IntervalGather( A->nnz, d_flagArray, d_index, A->nnz, d_tempArray, A->d_dcscRowInd, context );
		if( DEBUG ) CudaCheckError();
		IntervalGather( A->nnz, d_flagArray, d_index, A->nnz, A->d_cscVal, A->d_dcscVal, context );
		if( DEBUG ) CudaCheckError();

	gpu_timer5.Stop();

		// Check results
	if( DEBUG )
	{
		printf("Number of partitions: %d\n", partNum);
		printf("Doing radix sort on lower %d bits\n", end_bit);
		printf("DCSC col length: %d\n", A->col_length);
		CudaCheckError();
		print_array_device("dcscRowInd", A->d_dcscRowInd, A->nnz);
		print_array_device("dcscVal", A->d_dcscVal, A->nnz);
		//print_array_device("dcscRowInd", A->d_dcscRowInd+h_offsets[1]-1, A->nnz);
		//print_array_device("dcscRowInd", A->d_dcscRowInd+h_offsets[2]-1, A->nnz);
		print_array_device("dcscColPtr_off", A->d_dcscColPtr_off, A->col_length);
		print_array_device("dcscColPtr_ind", A->d_dcscColPtr_ind, A->col_length);
		CudaCheckError();
		print_array_device("dcscPartPtr", A->d_dcscPartPtr, partNum+1);
		CudaCheckError();
	}

		cudaFree(d_temp_storage);
		cudaFree(d_flagArray);
		cudaFree(d_tempArray);
		cudaFree(d_index);

		elapsed1 += gpu_timer1.ElapsedMillis();
		elapsed2 += gpu_timer2.ElapsedMillis();
		elapsed3 += gpu_timer3.ElapsedMillis();
		elapsed4 += gpu_timer4.ElapsedMillis();
		elapsed5 += gpu_timer5.ElapsedMillis();
		printf("Step 1: %f\n", elapsed1);
		printf("Step 2: %f\n", elapsed2);
		printf("Step 3: %f\n", elapsed3);
		printf("Step 4: %f\n", elapsed4);
		printf("Step 5: %f\n", elapsed5);
	}
}

void copy_part( d_matrix *A )
{
	cudaMemcpy( A->h_dcscPartPtr, A->d_dcscPartPtr, (A->part+1)*sizeof(int), cudaMemcpyDeviceToHost);
}
