#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <moderngpu.cuh>

typedef struct matrix{
	static const int DCSC = 32;

	//DCSC cudaMallocs
	int *d_dcscPartPtr;    // part+1
	int *d_dcscColPtr_ind; // O(n) in expectation: use DCSC*min(m,n) to be safe
	int *d_dcscColPtr_off; // O(n) in expectation: use DCSC*min(m,n) to be safe
	int *d_dcscRowInd;     // nnz
	float *d_dcscVal;      // nnz)

	//Device cudaMallocs
	int *d_cscColPtr;
	int *d_cscRowInd;
	float *d_cscVal;

	//DCSC host mallocs
	int *h_dcscPartPtr;

	//Host mallocs
	int *h_cscColPtr;
	int *h_cscRowInd;
	float *h_cscVal;

	int m;
	int n;
	int nnz;
	int part; 		// number of partitions
	int col_length; // precise length of d_dcscColPtr_ind and _off
} d_matrix;

void matrix_new( d_matrix *A, int m, int n );
void matrix_delete( d_matrix *A );
template<typename typeVal>
void buildMatrix( d_matrix *A, int numEdge, int *h_cooRowInd, int *h_cooColInd,
		                    typeVal *h_cooVal );
void matrix_copy( d_matrix *B, d_matrix *A );
void print_matrix( d_matrix *A, bool val=false );
void copy_matrix_device( d_matrix *A );
void print_matrix_device( d_matrix *A, bool val=false );
template<typename typeVal>
void extract( d_matrix *B, const d_matrix *A );
template<typename typeVal>
void extract_csr2csc( d_matrix *B, const d_matrix *A );
template <typename typeVal>
void csr_to_dcsc( d_matrix *A, int partSize, int partNum, mgpu::CudaContext& context, bool alloc=false );
void copy_part( d_matrix *A );

#endif
