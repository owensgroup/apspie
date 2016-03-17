#ifndef APSPIE_SRC_SCRATCH_HPP
#define APSPIE_SRC_SCRATCH_HPP

//template <typename T>
typedef struct scratch{
    //Device cudaMallocs
    int *d_cscVecInd;
    int *d_cscSwapInd;
    float *d_cscVecVal;
    float *d_cscSwapVal;
    float *d_cscTempVal;

    int *d_cscColGood;
    int *d_cscColBad;
    int *d_cscColDiff;
    int *d_ones;
    int *d_index;
    void *d_temp_storage;
    int *d_randVecInd;

    //Host mallocs
    int *h_cscVecInd;
    float *h_cscVecVal;
    int *h_cscColDiff;
    int *h_ones;
    int *h_index;

    int *h_bfsResult;
    float *h_spmvResult;
    float *h_bfsValA;
} d_scratch;

#endif  // APSPIE_SRC_SCRATCH_HPP
