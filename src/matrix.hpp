#ifndef MATRIX_HPP
#define MATRIX_HPP

typedef struct matrix{
	//DCSC cudaMallocs
	int *d_dcscPartPtr;    // O(part)
	int *d_dcscColPtr_ind; // O(n) in expectation
	int *d_dcscColPtr_off; // O(n) in expectation
	int *d_dcscRowInd;     // O(nnz)
	float *d_dcscVal;      // O(nnz)

	//Device cudaMallocs
	int *d_cscColPtr;
	int *d_cscRowInd;
	float *d_cscVal;

	//Host mallocs
	int *h_cscColPtr;
	int *h_cscRowInd;
	float *h_cscVal;

	int part;
	int m;
	int n;
	int nnz;
} d_matrix;

#endif
