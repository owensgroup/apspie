#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300
#define THREADS 1024
#define BLOCKS 256
#define LOG_THREADS 10
#define LOG_BLOCKS 8
#define CTA_OCCUPANCY 1
#define SHARED 1536

__device__ static const char logtable[256] =
{
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

__device__ int BinarySearch(int* keys, int count, int key) {
    int begin = 0;
    int end = count;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        int item = keys[mid];
        if (item == key) return 1;
        bool larger = (item > key);
        if (larger) end = mid;
        else begin = mid+1;
        }
        return 0;
}

__device__ unsigned ilog2(unsigned int v)
{
    register unsigned int t, tt;
    if (tt = v >> 16)
        return ((t = tt >> 8) ? 24 + logtable[t] : 16 + logtable[tt]);
    else 
        return ((t = v >> 8) ? 8 + logtable[t] : logtable[v]);
}

__device__ void deviceIntersectTwoSmallNL(
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_output_counts,
        int *d_output_total,
		const int stride)
    {
		__shared__ float block[SHARED];	
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        int start = threadIdx.x + blockIdx.x * blockDim.x;
        //int end = (start + stride * THREADS > input_length)? input_length :
        //                (start + stride * THREADS);
        //typedef cub::BlockReduce<int, THREADS> BlockReduceT;
        //__shared__ typename BlockReduceT::TempStorage temp_storage;

        for (int idx = start; idx < A->nnz; idx += BLOCKS*THREADS) {
            int count = 0;
            // get nls start and end index for two ids
            int sid = __ldg(A->d_cscRowInd+idx);
            //int sid = __ldg(d_src_node_ids+idx);
            int did = __ldg(A->d_cscRowInd+idx);
            int src_it = __ldg(A->d_cscColPtr+sid);
            int src_end = __ldg(A->d_cscColPtr+sid+1);
            int dst_it = __ldg(B->d_cscColPtr+did);
            int dst_end = __ldg(B->d_cscColPtr+did+1);
            if (src_it == src_end || dst_it == dst_end) continue;
            int src_nl_size = src_end - src_it;
            int dst_nl_size = dst_end - dst_it;
            int min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
            int max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
            int total = min_nl + max_nl;
            int *d_column_indices = (src_nl_size < dst_nl_size) ? B->d_cscRowInd : A->d_cscRowInd;
            if ( min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl ) {
                // search
                int min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
                int min_end = min_it + min_nl;
                int max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
                int *keys = &d_column_indices[max_it];
                //printf("src:%d,dst:%d, src_it:%d, dst_it:%d, min_it:%d max_it:%d, min max nl size: %d, %d\n",sid, did, src_it, dst_it, min_it, max_it, min_nl, max_nl);
                while ( min_it < min_end) {
                    int small_edge = d_column_indices[min_it++];
                    count += BinarySearch(keys, max_nl, small_edge);
                }
            } else {
                int src_edge = __ldg(d_column_indices+src_it);
                int dst_edge = __ldg(d_column_indices+dst_it);
                while (src_it < src_end && dst_it < dst_end) {
                    int diff = src_edge - dst_edge;
                    src_edge = (diff <= 0) ? __ldg(d_column_indices+(++src_it)) : src_edge;
                    dst_edge = (diff >= 0) ? __ldg(d_column_indices+(++dst_it)) : dst_edge;
                    count += (diff == 0);
                }
            }
            d_output_total[idx] += total;
            d_output_counts[idx] += count;
        }
    }
//};

/**
 * @brief Kernel entry for IntersectTwoSmallNL function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of int to the row offsets queue
 * @param[in] d_column_indices  Device pointer of int to the column indices queue
 * @param[in] A->d_cscRowInd    Device pointer of int to the incoming frontier queue (source node ids)
 * @param[in] B->d_cscRowInd    Device pointer of int to the incoming frontier queue (destination node ids)
 * @param[in] d_edge_list       Device pointer of int to the edge list IDs
 * @param[in] d_degrees         Device pointer of int to degree array
 * @param[in] problem           Device pointer to the problem object
 * @param[out] d_output_counts  Device pointer to the output counts array
 * @param[in] input_length      Length of the incoming frontier queues (A->d_cscRowInd and B->d_cscRowInd should have the same length)
 * @param[in] num_vertex        Maximum number of elements we can place into the incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the outgoing frontier
 *
 */

  template<typename typeVal>
  __launch_bounds__ (THREADS, CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_output_counts,
        int *d_output_total,
		const int stride)
{
    deviceIntersectTwoSmallNL( C, A, B, d_output_counts, d_output_total, stride );
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( 
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_output_counts,
        int *d_output_total,
        mgpu::CudaContext                             &context)
{
    int stride = (A->nnz + BLOCKS * THREADS - 1)
                        >> (LOG_THREADS + LOG_BLOCKS);
    
    IntersectTwoSmallNL<typeVal>
    <<<BLOCKS, THREADS>>>(
			C, A, B,
            d_output_counts,
            d_output_total,
            stride);

    long total = mgpu::Reduce(d_output_total, A->nnz, context);
    long tc_count = mgpu::Reduce(d_output_counts, A->nnz, context);
	print_array_device( d_output_total, A->nnz );
	print_array_device( d_output_counts, A->nnz );
    printf("tc_total:%ld, tc_count:%ld\n", total, tc_count);
    return (float)tc_count / (float)total;
    //return total_counts[0];
}

