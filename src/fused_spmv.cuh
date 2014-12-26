#include <moderngpu.cuh>

using namespace mgpu;

__global__ void GetNeighborListLen(const int *d_csr_row_ptr, int *d_neighbor_len, int *d_interval_value, const int node) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < node; idx += STRIDE) {
        d_neighbor_len[idx] = d_csr_row_ptr[idx+1] - d_csr_row_ptr[idx];
        d_interval_value[idx] = idx;
    }
}


template<typename T>
__global__ void ComputeResult(const T *d_csr_val, const int *d_csr_col_idx, const T *d_input_vector, const int *d_node_id, T *d_result, const int edge) {
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < edge; idx += STRIDE) {
        int index = d_csr_col_idx[idx];
        int r_idx = d_node_id[idx];
        if (d_input_vector[index] > 0) d_result[r_idx] = 1;
        //d_result[r_idx] = d_input_vector[index] > 0 ? 1 : d_result[r_idx];
        //if (d_input_vector[index] > 0) printf("%d:%f\n", r_idx, d_result[r_idx]);
    }
}
