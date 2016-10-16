#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300
#define THREADS 1024
#define BLOCKS 256
#define LOG_THREADS 10
#define LOG_BLOCKS 8
#define CTA_OCCUPANCY 1
#define SHARED 1536

#define CUDA_ERROR_CHECK
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

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

template< typename SizeT >
__device__ void deviceIntersectTwoSmallNL(
		d_matrix *C, 
		int *d_cscColPtrA,
		int *d_cscRowIndA,
		float *d_cscValA,
		int *d_cscColPtrB,
		int *d_cscRowIndB,
		float *d_cscValB, 
        int *d_output_counts,
        float *d_output_total,
		const SizeT m,
		const int partSize,
		const int stride)
    {
		//__shared__ float block[THREADS];	
		__shared__ float block[SHARED];	
        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        SizeT start = threadIdx.x + blockIdx.x * blockDim.x;
		SizeT tid = threadIdx.x;

        for (SizeT idx = start; idx < partSize*m; idx += stride) {
        //for (SizeT idx = start; idx < m*m; idx += stride) {
			int idx_row = idx/m;//partSize;
			int idx_col = idx%m;//%partSize;
			//printf("idx:%d, m:%lld, idx_row:%d, idx_col:%d\n", idx, m, idx_row, idx_col);

			int block_row = tid%THREADS;
            int count = 0;
			float sum = 0.0;
            // get nls start and end index for two ids
            int src_it = __ldg(d_cscColPtrA+idx_row);
            int src_end = __ldg(d_cscColPtrA+idx_row+1);
            int dst_it = __ldg(d_cscColPtrB+idx_col);
            int dst_end = __ldg(d_cscColPtrB+idx_col+1);
			/*int src_it = d_cscColPtrA[idx_row];
			int src_end = d_cscColPtrA[idx_row+1];
			int dst_it = d_cscColPtrB[idx_col];
			int dst_end = d_cscColPtrB[idx_col+1];*/
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);

            if (src_it == src_end || dst_it == dst_end) continue;
            int src_nl_size = src_end - src_it;
            int dst_nl_size = dst_end - dst_it;
            int min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
            int max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
            if ( min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl ) {
                // search
            	int *d_column_indices = (src_nl_size < dst_nl_size) ? d_cscRowIndB  : d_cscRowIndA;
                int min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
                int min_end = min_it + min_nl;
                int max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
                int *keys = &d_column_indices[max_it];
                while ( min_it < min_end) {
                    int small_edge = d_column_indices[min_it++];
                    count += BinarySearch(keys, max_nl, small_edge);
					//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, small_edge: %d\n", idx, src_it, src_end, dst_it, dst_end, small_edge);
                }
            } else {
                //int src_edge = __ldg(d_cscRowIndA+src_it);
                //int dst_edge = __ldg(d_cscRowIndB+dst_it);
                int src_edge = d_cscRowIndA[src_it];
                int dst_edge = d_cscRowIndB[dst_it];
                while (src_it < src_end && dst_it < dst_end) {
                    int diff = src_edge - dst_edge;
					sum += (diff == 0) ? __ldg(d_cscValA+src_it)*__ldg(d_cscValB+dst_it) : 0;
					//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, src_edge:%d, dst_edge:%d\n", idx, src_it, src_end, dst_it, dst_end, src_edge, dst_edge);
                    src_edge = (diff <= 0) ? __ldg(d_cscRowIndA+(++src_it)) : src_edge;
                    dst_edge = (diff >= 0) ? __ldg(d_cscRowIndB+(++dst_it)) : dst_edge;
					/*sum += (diff == 0) ? d_cscValA[src_it]*d_cscValB[dst_it] : 0;
					//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, src_edge:%d, dst_edge:%d\n", idx, src_it, src_end, dst_it, dst_end, src_edge, dst_edge);
                    src_edge = (diff <= 0) ? d_cscRowIndA[++src_it] : src_edge;
                    dst_edge = (diff >= 0) ? d_cscRowIndB[++dst_it] : dst_edge;*/
                    count += (diff == 0);
                }
            }
			//block[block_row]  = sum;
			//printf("blk_row:%d, idx:%d, val:%f\n", block_row, idx, sum );
			if( sum > 0.001 )
				d_output_total[idx_col] = sum;
            //d_output_counts[idx] = src_end;
            //d_output_counts[idx_col] += count;
    	}
}

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

  //__launch_bounds__ (THREADS, CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
		d_matrix *C,
		int *d_cscColPtrA,
		int *d_cscRowIndA,
		float *d_cscValA,
		int *d_cscColPtrB,
		int *d_cscRowIndB,
		float *d_cscValB, 
        int *d_output_counts,
        float *d_output_total,
		const int m,
		const int partSize,
		const int stride)
{
    deviceIntersectTwoSmallNL<long long>( C, 
		d_cscColPtrA,
		d_cscRowIndA,
		d_cscValA,
		d_cscColPtrB,
		d_cscRowIndB,
		d_cscValB, 
		d_output_counts, d_output_total, m, partSize, stride );
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( 
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_output_counts,
        float *d_output_total,
		const int partSize,
        mgpu::CudaContext                             &context)
{
	//const int BLOCKS = (A->m*A->m+THREADS-1)/THREADS;
    int stride = BLOCKS * THREADS;
    //int stride = (A->nnz + BLOCKS * THREADS - 1)
    //                    >> (LOG_THREADS + LOG_BLOCKS);

    // Set 48kB shared memory 
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferShared));
   
	GpuTimer gpu_timer;
	float elapsed = 0.0f;
	gpu_timer.Start();

    IntersectTwoSmallNL<<<BLOCKS, THREADS>>>(
    //IntersectTwoSmallNL<<<1, A->m*A->m>>>(
    //IntersectTwoSmallNL<<<BLOCKS, THREADS>>>(
			C,
			A->d_cscColPtr,
			A->d_cscRowInd,
			A->d_cscVal,
			B->d_cscColPtr,
			B->d_cscRowInd,
			B->d_cscVal, 
            d_output_counts,
            d_output_total,
			(long long) A->m,
			partSize,
            stride);

	gpu_timer.Stop();
	elapsed += gpu_timer.ElapsedMillis();
	printf("my spgemm: %f ms\n", elapsed);
	CudaCheckError();

    float total = mgpu::Reduce(d_output_total, A->m, context);
    long tc_count = mgpu::Reduce(d_output_counts, A->m, context);
	print_array_device( d_output_total, A->m );
	print_array_device( d_output_counts, A->m );
    printf("tc_total:%f, tc_count:%ld\n", total, tc_count);
    return (float)tc_count / (float)total;
}

