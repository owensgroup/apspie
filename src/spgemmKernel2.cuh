#include <cub/cub.cuh>
#include <cuda_profiler_api.h>

#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300
#define THREADS 128
#define BLOCKS 480  // 32*NUM_SM
//#define BLOCKS 1
#define LOG_THREADS 10
#define LOG_BLOCKS 8
#define CTA_OCCUPANCY 1
#define SHARED 1024
#define UNROLL 1  // SHARED/THREADS
#define MAX_PART 200
#define BLOCK_SIZE 1024

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

__constant__ int nnzPartA[MAX_PART];
__constant__ int nnzPartB[MAX_PART];
__constant__ int numBlockA[MAX_PART];
__constant__ int numBlockB[MAX_PART];

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
__device__ int deviceIntersectTwoSmallNL(
		int *d_cscColIndC,
		int *d_cscRowIndC,
		float *d_cscValC,
		int d_numInter,        // d_inter[i+1]-d_inter[i]
		int *d_intertarget,
		int d_numPartPtrA,     // single int representing offset for ColPtr
		int *d_intersectionA,
		int *d_dcscColPtr_indA,
		int *d_dcscColPtr_offA,
		int *d_dcscRowIndA,
		float *d_dcscValA,
		int d_numPartPtrB,     // single int representing offset for ColPtr
		int *d_intersectionB,
		int *d_dcscColPtr_indB,
		int *d_dcscColPtr_offB,
		int *d_dcscRowIndB,
		float *d_dcscValB, 
		const int stride)
{
		__shared__ cub::BlockReduce<int, THREADS>::TempStorage temp;
		__shared__ cub::BlockReduce<float, THREADS>::TempStorage tempFloat;

        // each thread process NV edge pairs
        // Each block get a block-wise intersect count
        SizeT gid = threadIdx.x + blockIdx.x * blockDim.x;
		SizeT tid = threadIdx.x;
		SizeT wid = gid>>5;
		

		// Use col_lengthA for now (for symmetric matrices)
        for (SizeT idx = wid; idx < d_numInter; idx += stride) {
        //for (SizeT idx = gid; idx < h_inter1; idx += stride) {
        //for (SizeT idx = gid; idx < partNum*partNum*col_lengthA; idx += stride) {

			/*#pragma unroll
			for( int idx_inner = 0; idx_inner<UNROLL; idx_inner++ )
			{
				tid_thread += THREADS;
				s_cscRowIndA[tid_thread] = d_cscRowIndA[block_A*SHARED+tid_thread];
				s_cscValA[tid_thread] = __ldg(d_cscValA+block_A*SHARED+tid_thread);
				s_cscRowIndB[tid_thread] = d_cscRowIndB[block_B*SHARED+tid_thread];
				s_cscValB[tid_thread] = __ldg(d_cscValB+block_B*SHARED+tid_thread);
			}*/

			int nodeA = d_intersectionA[idx];
			int nodeB = d_intersectionB[idx];

			//int write = d_intertarget[idx];

            // get nls start and end index for two ids
            int src_it = __ldg(d_dcscColPtr_offA+nodeA+d_numPartPtrA);
            int src_end = __ldg(d_dcscColPtr_offA+nodeA+d_numPartPtrA+1);
            int dst_it = __ldg(d_dcscColPtr_offB+nodeB+d_numPartPtrB);
            int dst_end = __ldg(d_dcscColPtr_offB+nodeB+d_numPartPtrB+1);
            //int src_it = __ldg(d_dcscColPtr_offA+idx);
            //int src_end = __ldg(d_dcscColPtr_offA+idx+1);
            //int dst_it = __ldg(d_dcscColPtr_offB+idx);
            //int dst_end = __ldg(d_dcscColPtr_offB+idx+1);
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);
			//printf("idx:%d, src_it:%d, src_end:%d, dst_it:%d, dst_end:%d, dst_end:%d, dst_end:%d\n", idx, src_it, src_end, dst_it, dst_end, dst_end, dst_end);

			int dst_safe = dst_it;
			int count = 0;
			int row_idx = 0;
			int col_idx = 0;
			float val1 = 0.0f;
			float val2 = 0.0f;

			for( SizeT idx2 = tid+src_it; idx2<src_end; idx2+=32 ) {
				col_idx += d_dcscRowIndA[idx2];
				val1  += d_dcscValA[idx2];
			}

			for( SizeT idx2 = tid+dst_it; idx2<dst_end; idx2+=32 ) {
				row_idx += d_dcscRowIndB[idx2];
				val2 += d_dcscValB[idx2];
			}
			/*	while( dst_it < dst_end ) {
					row_idx += d_dcscRowIndB[dst_it];
					val2  += d_dcscValB[dst_it];
					//d_cscColIndC[write] = col_idx;
					//d_cscRowIndC[write] = row_idx;
					//d_cscValC[write] = val1*val2;
					write++;
					dst_it++;
					count++;
				}*/

			int result;
			int result2;
			float resultFloat;
    		if(tid<THREADS) {
				result = cub::BlockReduce<int, THREADS>(temp).Sum(row_idx);
    			result2 = cub::BlockReduce<int, THREADS>(temp).Sum(col_idx);
				resultFloat = cub::BlockReduce<float, THREADS>(tempFloat).Sum(val1);
				resultFloat += cub::BlockReduce<float, THREADS>(tempFloat).Sum(val2);
			}
			if(tid==0) { 
				//printf("Total: %d\n", result); 
				d_cscColIndC[blockIdx.x] = result;
				d_cscRowIndC[blockIdx.x] = result2;
				d_cscValC[blockIdx.x] = resultFloat;
				return result; 
			}
    }
}

/*
 * @brief Kernel entry for IntersectTwoSmallNL function
 */

  //__launch_bounds__ (THREADS, CTA_OCCUPANCY)
  __global__
  void IntersectTwoSmallNL(
		int *d_cscColIndC,
		int *d_cscRowIndC,
		float *d_cscValC,
		int *d_inter,
		int *d_intertarget,
		int *d_dcscPartPtrA,
		int *d_intersectionA,
		int *d_dcscColPtr_indA,
		int *d_dcscColPtr_offA,
		int *d_dcscRowIndA,
		float *d_dcscValA,
		int *d_dcscPartPtrB,
		int *d_intersectionB,
		int *d_dcscColPtr_indB,
		int *d_dcscColPtr_offB,
		int *d_dcscRowIndB,
		float *d_dcscValB, 
		const int partNum,
		const int stride)
{
	for( int i=0; i<partNum; i++ )
		for( int j=0; j<partNum; j++ )
	//for( int i=0; i<partNum; i++ )
    //	for( int j=0; j<1; j++ )
	int length = deviceIntersectTwoSmallNL<long long>( 
			d_cscColIndC,
			d_cscRowIndC,
			d_cscValC,
			d_inter[i*partNum+j+1]-d_inter[i*partNum+j],
			d_intertarget,
			d_dcscPartPtrA[i],
			d_intersectionA+d_inter[i*partNum+j],
			d_dcscColPtr_indA,
			d_dcscColPtr_offA,
			d_dcscRowIndA,
			d_dcscValA,
			d_dcscPartPtrB[j],
			d_intersectionB+d_inter[i*partNum+j],
			d_dcscColPtr_indB,
			d_dcscColPtr_offB,
			d_dcscRowIndB,
			d_dcscValB, 
            stride);
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( 
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_intersectionA,
        int *d_intersectionB,
        int *d_interbalance,
		int *h_inter,
		const int maxAB,
		const int partSize,
		const int partNum,
        mgpu::CudaContext                             &context)
{
	//const int BLOCKS = (A->m*A->m+THREADS-1)/THREADS;
	//int mEven = (A->m+THREADS-1)/THREADS;
    int stride = BLOCKS * THREADS;

    // Set 48kB shared memory 
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferL1));
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(IntersectTwoSmallNL, cudaFuncCachePreferShared));
   
	// Need segmented sort
    int *d_inter;
    cudaMalloc(&d_inter, (partNum*partNum+1)*sizeof(int));
    cudaMemcpy(d_inter, h_inter, (partNum*partNum+1)*sizeof(int), cudaMemcpyHostToDevice);

	int *d_intertarget;
    cudaMalloc(&d_intertarget, maxAB*sizeof(int));

	CudaCheckError();
	cudaProfilerStart();
	GpuTimer gpu_timer;
	float elapsed = 0.0f;
	gpu_timer.Start();

	int length = 0;
	//mgpu::Scan<mgpu::MgpuScanTypeExc>( d_interbalance, h_inter[(i+1)*partNum], 0, mgpu::plus<int>(), (int*)0, &(length), d_intertarget, context );
		//print_array_device("intertarget", d_intertarget, h_inter[(i+1)*partNum]);
	printf("Length:%d\n", length);
    IntersectTwoSmallNL<<<BLOCKS, THREADS>>>(
			C->d_cscColInd,
			C->d_cscRowInd,
			C->d_cscVal,
			d_inter,
			d_intertarget,
			A->d_dcscPartPtr,
			d_intersectionA,
			A->d_dcscColPtr_ind,
			A->d_dcscColPtr_off,
			A->d_dcscRowInd,
			A->d_dcscVal,
			B->d_dcscPartPtr,
			d_intersectionB,
			B->d_dcscColPtr_ind,
			B->d_dcscColPtr_off,
			B->d_dcscRowInd,
			B->d_dcscVal,
			partNum, 
            stride);

	gpu_timer.Stop();
	cudaProfilerStop();
	elapsed += gpu_timer.ElapsedMillis();
	printf("my spgemm: %f ms\n", elapsed);
	CudaCheckError();

	print_array("h_inter", h_inter, partNum);
	print_array_device("Off", A->d_dcscColPtr_off+h_inter[1], 40);
	print_array_device("Row", A->d_cscColInd, h_inter[1]);
	print_array_device("Col", A->d_cscRowInd, h_inter[1]);
	print_array_device("Val", A->d_cscVal, h_inter[1]);

}

