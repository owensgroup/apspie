#ifndef SCRATCH_HPP
#define SCRATCH_HPP

//template <typename T>
typedef struct scratch{
    //Device cudaMallocs
    int *d_csrVecInd;
    int *d_csrSwapInd;
    float *d_csrVecVal;
    float *d_csrSwapVal;
    float *d_csrTempVal;

    int *d_csrRowGood;
    int *d_csrRowBad;
    int *d_csrRowDiff;
    int *d_ones;
    int *d_index;
    void *d_temp_storage;
    int *d_randVecInd;

    //Host mallocs
    int *h_csrVecInd;
    int *h_csrVecVal;
    int *h_csrRowDiff;
    int *h_ones;
    int *h_index;

    int *h_bfsResult;
    float *h_spmvResult;
    float *h_bfsValA;
} d_scratch;

#endif
