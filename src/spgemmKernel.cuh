#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300
#define THREADS 1024
#define BLOCKS 480
//#define BLOCKS 1
#define LOG_THREADS 10
#define LOG_BLOCKS 8
#define CTA_OCCUPANCY 1
#define SHARED 1024

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

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

__device__ int BinarySearchStart(int* keys, int count, int key) {
    int begin = 0;
    int end = count;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        int item = keys[mid];
        if (item == key) return mid;
        bool larger = (item > key);
        if (larger) end = mid;
        else begin = mid+1;
    }
    return end-1;
}

__device__ int BinarySearchEnd(int* keys, int count, int key) {
    int begin = 0;
    int end = count;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        int item = keys[mid];
        if (item == key) return mid;
        bool larger = (item > key);
        if (larger) end = mid;
        else begin = mid+1;
    }
    return end;
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
		const int m,
		const int nnz,
		const int partSize,
		const int partNum,
		const int maxBlockA,
		const int maxBlockB,
		const int *nnzPartA,
		const int *nnzPartB,
		const int *numBlockA,
		const int *numBlockB,
		const int stride)
    {
		//__shared__ float block[THREADS];	
		__shared__ int s_cscColPtrA[SHARED];	
		__shared__ int s_cscRowIndA[SHARED];	
		__shared__ float s_cscValA[SHARED];	
		__shared__ int s_cscColPtrB[SHARED];	
		__shared__ int s_cscRowIndB[SHARED];	
		__shared__ float s_cscValB[SHARED];
		__shared__ int s_cscColPtrA_bound[2]; //[0]: start, [1]: end
		__shared__ int s_cscColPtrB_bound[2]; //[0]: start, [1]: end
		__shared__ int s_nnzPart[2];		  //[0]: A,		[1]: B
		__shared__ int s_numBlock[2];		  //[0]: A,		[1]: B
		__shared__ int s_cscColPtr_begin;		  //[0]: A,		[1]: B

        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        SizeT start = threadIdx.x + blockIdx.x * blockDim.x;
		SizeT tid = threadIdx.x;
        SizeT gid = start-tid;

		int maxBlockAB = maxBlockA*maxBlockB;
		//printf("idx:%d, nBlockA:%d, nBlockB:%d\n", start, numBlockA, numBlockB);
        for (SizeT idx = (long long) maxBlockAB*SHARED; idx < (long long) (10+maxBlockAB)*SHARED; idx += (long long)stride) {
        //for (SizeT idx = start; idx < (long long) maxBlockAB*SHARED; idx += stride) {
        //for (SizeT idx = start; idx < (long long) partNum*partNum*maxBlockA*maxBlockB*SHARED; idx += stride) {
			int part_A=(maxBlockAB+blockIdx.x)/maxBlockAB;
			int part_B=(maxBlockAB+blockIdx.x)%maxBlockAB;

			int rank_A=(maxBlockAB+blockIdx.x) - part_A;
			int rank_B=(maxBlockAB+blockIdx.x) - part_B;

			if( tid==0 ) s_nnzPart[0] = __ldg( nnzPartA+part_A );
			if( tid==1 ) s_nnzPart[1] = __ldg( nnzPartB+part_B );
			if( tid==2 ) s_numBlock[0]= __ldg( numBlockA+part_A );
			if( tid==3 ) s_numBlock[1]= __ldg( numBlockB+part_B );
			if( tid==4 ) s_cscColPtr_begin=__ldg(d_cscColPtrA+part_A*partSize);

			//int idx_A = rank_A+tid;
			//int idx_B = idx%s_numBlock[1];

			if( rank_A >= s_numBlock[0] || rank_B >=s_numBlock[1] ) continue; 
			printf("idx:%d, partA:%d, partB:%d, rankA:%d, rankB:%d, rankB:%d\n", idx, part_A, part_B, rank_A, rank_B, rank_B);
			printf("idx:%d, i:%d, j:%d, blockA:%d, blockB:%d, begin: %d, begin:%d, idx_B:%d, idx_B:%d\n", idx, part_A, part_B, s_numBlock[0], s_numBlock[1], s_cscColPtr_begin, s_cscColPtr_begin);

			/*int idx_A = idx/s_numBlock[1]*SHARED+tid;
			int idx_B = idx%s_numBlock[1];
			//if( idx<128 ) printf("idx:%d, i:%d, j:%d, begin: %d, idx_A:%d, idx_B:%d, idx_B:%d\n", idx, i, j, begin_ColPtr, idx_A, idx_B, idx_B);
			if( idx_A < s_nnzPart[0] )
			{
				s_cscRowIndA[tid] = __ldg(d_cscRowIndA+idx_A+s_cscColPtr_begin);
				s_cscValA[tid] = __ldg(d_cscValA+idx_A+s_cscColPtr_begin);
			}
			//s_cscRowIndB[tid] = __ldg(d_cscRowIndB+idx);
			//s_cscValB[tid] = __ldg(d_cscValB+idx);

			// Generate cscColPtrA that is local to cscRowIndA[0 ... 1023]
			if( tid==0 )
				s_cscColPtrA_bound[0] = BinarySearchStart( d_cscColPtrA, m, gid );
			if( tid==1 )
				s_cscColPtrA_bound[1] = BinarySearchEnd( d_cscColPtrA, m, gid+blockDim.x-1 );
			//int s_cscColPtrB_start = BinarySearchStart( d_cscColPtrB, m, gid );
			//int s_cscColPtrB_end = BinarySearchEnd( d_cscColPtrB, m, gid+blockDim.x-1 );*/

			//__syncthreads();

			/*int block_row = tid%THREADS;
            int count = 0;
			float sum = 0.0;
            // get nls start and end index for two ids
            int src_it = __ldg(d_cscColPtrA+idx_row);
            int src_end = __ldg(d_cscColPtrA+idx_row+1);
            int dst_it = __ldg(d_cscColPtrB+idx_col);
            int dst_end = __ldg(d_cscColPtrB+idx_col+1);
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);

            if (src_it == src_end || dst_it == dst_end) continue;
            int src_nl_size = src_end - src_it;
            int dst_nl_size = dst_end - dst_it;
            int min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
            int max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
            if ( min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl ) {
                // search
            	//int *d_column_indices = (src_nl_size < dst_nl_size) ? d_cscRowIndB  : d_cscRowIndA;
            	int *d_column_indices = (src_nl_size < dst_nl_size) ? s_cscRowIndB  : s_cscRowIndA;
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
                int src_edge = s_cscRowIndA[src_it];
                int dst_edge = s_cscRowIndB[dst_it];
                while (src_it < src_end && dst_it < dst_end) {
                    int diff = src_edge - dst_edge;
					sum += (diff == 0) ? s_cscValA[src_it]*s_cscValB[dst_it] : 0;
                    src_edge = (diff <= 0) ? s_cscRowIndA[++src_it] : src_edge;
                    dst_edge = (diff >= 0) ? s_cscRowIndB[++dst_it] : dst_edge;
					//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, src_edge:%d, dst_edge:%d\n", idx, src_it, src_end, dst_it, dst_end, src_edge, dst_edge);
					//sum += (diff == 0) ? __ldg(d_cscValA+src_it)*__ldg(d_cscValB+dst_it) : 0;
                    //src_edge = (diff <= 0) ? __ldg(d_cscRowIndA+(++src_it)) : src_edge;
                    //dst_edge = (diff >= 0) ? __ldg(d_cscRowIndB+(++dst_it)) : dst_edge;
                    count += (diff == 0);
                }
            }
			//if( sum > 0.001 )

			//printf("blk_row:%d, idx:%d, val:%f\n", block_row, idx, sum );
			d_output_counts[idx_B] = s_cscRowIndA[tid];
			d_output_counts[idx_B] = s_cscRowIndB[tid];
			d_output_total[idx_B] = s_cscRowIndA[tid];
			d_output_total[idx_B] = s_cscRowIndB[tid];*/
    	}
}

/*
 * @brief Kernel entry for IntersectTwoSmallNL function
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
		const int nnz,
		const int partSize,
		const int partNum,
		const int maxBlockA,
		const int maxBlockB,
		int *d_nnzPartA,
		int *d_nnzPartB,
		int *d_numBlockA,
		int *d_numBlockB,
		const int stride)
{
	//for( int i=0; i<partNum; i++ )
	//	for( int j=0; j<partNum; j++ )
	//for( int i=0; i<partNum; i++ )
    //	for( int j=0; j<1; j++ )
	deviceIntersectTwoSmallNL<long long>( C, 
		d_cscColPtrA,
		d_cscRowIndA,
		d_cscValA,
		d_cscColPtrB,
		d_cscRowIndB,
		d_cscValB, 
		d_output_counts, d_output_total, m, nnz, partSize, partNum, 
		maxBlockA,
		maxBlockB,
		d_nnzPartA,
		d_nnzPartB,
		d_numBlockA,
		d_numBlockB,
		stride );
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( 
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_output_counts,
        float *d_output_total,
		const int partSize,
		const int partNum,
		const int maxBlockA,
		const int maxBlockB,
		int *d_nnzPartA,
		int *d_nnzPartB,
		int *d_numBlockA,
		int *d_numBlockB,
        mgpu::CudaContext                             &context)
{
	//const int BLOCKS = (A->m*A->m+THREADS-1)/THREADS;
	//int mEven = (A->m+THREADS-1)/THREADS;
    int stride = BLOCKS * THREADS;

    // Set 48kB shared memory 
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferShared));
   
	CudaCheckError();
	GpuTimer gpu_timer;
	float elapsed = 0.0f;
	gpu_timer.Start();

    //IntersectTwoSmallNL<<<1, THREADS>>>(
    IntersectTwoSmallNL<<<BLOCKS, THREADS>>>(
			C,
			A->d_cscColPtr,
			A->d_cscRowInd,
			A->d_cscVal,
			B->d_cscColPtr,
			B->d_cscRowInd,
			B->d_cscVal, 
            d_output_counts,
            d_output_total,
			A->m,
			A->nnz,
			partSize,
			partNum,
			maxBlockA,
			maxBlockB,
			d_nnzPartA,
			d_nnzPartB,
			d_numBlockA,
			d_numBlockB,
            stride);

	gpu_timer.Stop();
	elapsed += gpu_timer.ElapsedMillis();
	printf("my spgemm: %f ms\n", elapsed);
	//CudaCheckError();

    float total = mgpu::Reduce(d_output_total, A->m, context);
    long tc_count = mgpu::Reduce(d_output_counts, A->m, context);
	print_array_device( d_output_total, A->m );
	print_array_device( d_output_counts, A->m );
    printf("tc_total:%f, tc_count:%ld\n", total, tc_count);
    return (float)tc_count / (float)total;
}

