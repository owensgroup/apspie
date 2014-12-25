//------------------------
//Quick and dirty spmv implementation
//------------------------

#include <moderngpu.cuh>
#include <kernels/spmvcsr.cuh>

using namespace mgpu;

template<typename T>
void SpmvKernel(const T *d_csr_val,
                 const int *d_csr_col_idx,
                 const int *d_row_ptr,
                 const T *d_input_vector,
                 T *d_results,
                 const int num_node,
                 const int num_edge,
                 CudaContext& context) { 
    SpmvCsrBinary(d_csr_val, d_csr_col_idx, num_edge, d_row_ptr, num_node, 
				d_input_vector, true, d_results, (T)0, mgpu::multiplies<T>(),
				mgpu::plus<T>(), context);

}
