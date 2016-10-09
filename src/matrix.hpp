#ifndef MATRIX_HPP
#define MATRIX_HPP

typedef struct matrix{
	//Device cudaMallocs
	int *d_cscColPtr;
	int *d_cscRowInd;
	float *d_cscVal;

	//Host mallocs
	int *h_cscColPtr;
	int *h_cscRowInd;
	float *h_cscVal;

	int m;
	int n;
	int nnz;
} d_matrix;

#endif
