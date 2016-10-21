#include <iomanip>
#include <cub/cub.cuh>
#include <mgpu/

#include "matrix.hpp"

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
	cudaMalloc(&(A->d_dcscColPtr_ind), 2*A->n*sizeof(int));
	cudaMalloc(&(A->d_dcscColPtr_off), 2*A->n*sizeof(int));
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
void csr_to_dcsc( d_matrix *A, int partSize, int partNum, bool alloc )
{
	if( alloc )
	{
		A->part = partNum;

		cudaMalloc(&(A->d_dcscPartPtr), (partNum+1)*sizeof(int));
	} else {
	// Step 0: Preallocations for scratchpad memory
	int *d_flagArray, *d_tempArray;
	CUDA_SAFE_CALL(cudaMalloc(&d_flagArray, A->nnz*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&d_tempArray, A->nnz*sizeof(int)));

	// Step 1: Segmented Sort RowInd of CSC 
	// Set up parameters
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		int *d_keys_in = A->d_cscRowInd;
		int *d_keys_out= A->d_dcscRowInd;
		int num_items = A->nnz;
		int num_segments = partNum;

		// CUDA mallocs
		int *h_offsets = (int*)malloc((partNum+1)*sizeof(int));
		for( int i=0; i<partNum; i++ ) h_offsets[i]=A->h_cscColPtr[i*partSize];
		h_offsets[partNum] = A->nnz;
		int *d_offsets;
		cudaMalloc(&d_offsets, (partNum+1)*sizeof(int));
		cudaMemcpy(d_offsets,h_offsets,(partNum+1)*sizeof(int),cudaMemcpyHostToDevice);

		// Determine temporary device storage requirements
		print_array_device(d_keys_out+A->nnz-40);
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments, d_offsets, d_offsets + 1);
		//CudaCheckError();

		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);

		// Run sorting operation
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, num_segments, d_offsets, d_offsets + 1);
		//CudaCheckError();

		// Check results
		print_array_device(d_keys_in);
		print_array_device(d_keys_out);
		cudaFree(d_temp_storage);

	// Step 2: SegReduce
	// Determine temporary device storage requirements
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		int *d_in = d_keys_out;
		d_offsets = A->d_dcscColPtr_ind;

		// Obtain d_offsets. In this case, d_offsets is actually our d_dcscColPtr_off
		lookRight<<<BLOCKS,THREADS>>>( d_flagArray, num_items, d_in );
		mgpu::Scan<mgpu::MgpuScanTypeExc>( d_flagArray, num_items, 0, mgpu::plus<int>(), (int*)0, &(A->col_length), d_tempArray, context );
		if( A->col_length > 2*A->n ) printf("Error: array too long\n");
	 	streamCompact<<<BLOCKS,THREADS>>>( d_flagArray, d_tempArray, A->d_dcscColPtr_ind, num_items );

		/*cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_offsets, d_offsets + 1, min_op, initial_value);

		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);

		// Run reduction
		cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_offsets, d_offsets + 1, min_op, initial_value);

		// Alternative SegReduce
		ReduceByKey( d_keys_in, d_vals_in, num_items, (float)0, mgpu::plus<int>(), mgpu::equal_to<int>(), d_keys_out, d_vals_out, &h_cscVecCount, &, context );*/
	}
}
