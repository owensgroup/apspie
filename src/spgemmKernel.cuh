#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include "../ext/moderngpu/include/device/ctaloadbalance.cuh"
#include <mgpudevice.cuh>
#include <mt19937ar.h>

#include "triple.hpp"

#define __GR_CUDA_ARCH__ 300
#define THREADS 128
#define BLOCKS 512  // 32*NUM_SM
#define STRIDE 2048
#define LOG_THREADS 10
#define LOG_BLOCKS 8
#define CTA_OCCUPANCY 1
#define SHARED 1024
#define UNROLL 1  // SHARED/THREADS
#define MAX_PART 200
#define BLOCK_SIZE 1024
#define TABLE_SIZE 131071
#define MAX_PROBES 200
#define SLOT_EMPTY 0xffffffff
#define SLOT_EMPTY_INIT 255
#define JUMP_HASH 41
#define PRIME_DIV 4294967291

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

__device__ unsigned mixHash(unsigned a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

void generateConstants( uint2 *constants ) {
	  unsigned new_a = genrand_int32() % PRIME_DIV;
      constants->x = (1 > new_a ? 1 : new_a);
      constants->y = genrand_int32() % PRIME_DIV;
}

__device__ bool insert( const int row_idx, const int col_idx, const float valC, unsigned *d_hashKey, float *d_hashVal, uint2 constants ) {
	unsigned key = (row_idx<<16) | (col_idx&65535);
	// Attempt 2a: Basic hash
	//unsigned hash = key & TABLE_SIZE;
    // Attempt 2b: Mix hash
	unsigned mix  = mixHash(key);
	//unsigned hash = mix & TABLE_SIZE;
    // Attempt 2c: Alcantara hash
	unsigned hash = ((constants.x ^ mix + constants.y ) % PRIME_DIV) & TABLE_SIZE;
	unsigned doubleHashJump = mix % JUMP_HASH + 1;

	for( int attempt = 1; attempt<=MAX_PROBES; attempt++) {
		//printf("key:%d, hash:%d, table:%d\n", key, hash, d_hashKey[hash]);
		if( d_hashKey[hash] == key ) {
			atomicAdd( d_hashVal+hash, valC );
			return true;
		}
		if( d_hashKey[hash] == SLOT_EMPTY ) {
			unsigned old = atomicCAS( d_hashKey+hash, SLOT_EMPTY, key );
			if( old==SLOT_EMPTY ) {
				atomicAdd( d_hashVal+hash, valC );
				return true;
			}
		}
		// Attempt 5a: Linear probing
		//hash = (hash+1) & TABLE_SIZE;
		// Attempt 5b: Quadratic probing
		//hash = (hash+attempt*attempt) & TABLE_SIZE;
		// Attempt 5c: Prime jump
		hash = (hash+attempt*doubleHashJump) & TABLE_SIZE;
	}
	return false;
}

template<typename Tuning, typename IndicesIt, typename ValuesIt, typename OutputIt>
__global__ void KernelInterval(int destCount,
    IndicesIt indices_global, ValuesIt values_globalA, ValuesIt d_offB, ValuesIt d_lengthB, IndicesIt d_dcscRowIndA, float *d_dcscValA, IndicesIt d_dcscRowIndB, float *d_dcscValB, int sourceCount, const int* mp_global, OutputIt output_global, unsigned *d_hashKey, float *d_hashVal, int *value, uint2 constants ) {

    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    typedef typename std::iterator_traits<ValuesIt>::value_type T;

    union Shared {
        int indices[NT * (VT + 1)];
        T values[NT * VT];
    };
    __shared__ Shared shared;
    int tid = threadIdx.x;
    int block = blockIdx.x;

    // Compute the input and output intervals this CTA processes.
    int4 range = mgpu::CTALoadBalance<NT, VT>(destCount, indices_global, sourceCount,
        block, tid, mp_global, shared.indices, true);

    // The interval indices are in the left part of shared memory (moveCount).
    // The scan of interval counts are in the right part (intervalCount).
    destCount = range.y - range.x;
    sourceCount = range.w - range.z;

    // Copy the source indices into register.
    int sources[VT];
    mgpu::DeviceSharedToReg<NT, VT>(shared.indices, tid, sources);

    // The interval indices are in the left part of shared memory (moveCount).
    // The scan of interval counts are in the right part (intervalCount).
    int* move_shared = shared.indices;
    int* intervals_shared = shared.indices + destCount;
    int* intervals_shared2 = intervals_shared - range.z;
 
    // Read out the interval indices and scan offsets.
    int interval[VT], rank[VT];
    #pragma unroll
    for(int i = 0; i < VT; ++i) {
        int index = NT * i + tid;
        int gid = range.x + index;
        interval[i] = range.z;
        if(index < destCount) {
            interval[i] = move_shared[index];
            rank[i] = gid - intervals_shared2[interval[i]];
        }
    }
    __syncthreads();

    // Load the source fill values into shared memory. Each value is fetched
    // only once to reduce latency and L2 traffic.
    mgpu::DeviceMemToMemLoop<NT>(sourceCount, values_globalA + range.z, tid,
        shared.values);

    // Gather the values from shared memory into register. This uses a shared
    // memory broadcast - one instance of a value serves all the threads that
    // comprise its fill operation.
    T values[VT];
    mgpu::DeviceGather<NT, VT>(destCount, shared.values - range.z, sources, tid,
        values, false);

	int gather[VT], length[VT], off[VT], row_idx[VT], col_idx[VT];
	float valA[VT], valB[VT], valC[VT];

    #pragma unroll
    for(int i = 0; i < VT; ++i) {
        int index = NT * i + tid;
        int gid = range.x + index;
		if( index < destCount ) {
        	gather[i] = values[i] + rank[i];
			length[i] = __ldg(d_lengthB+sources[i]);    // lengthB
			off[i]   = __ldg(d_offB+sources[i]);        // offB
			row_idx[i]= __ldg(d_dcscRowIndA+gather[i]);
			valA[i]   = __ldg(d_dcscValA+gather[i]);
			//if( tid<32 ) printf("tid:%d, vid:%d, gather: %d, off:%d, length:%d, row:%d, val:%f\n", tid, i, gather[i], off[i], length[i], row_idx[i], valA[i]); 

			for( int j=0; j<length[i]; j++ ) {
				col_idx[i] = __ldg(d_dcscRowIndB+off[i]+j);
				valB[i]    = __ldg(d_dcscValB+off[i]+j);
				valC[i]    = valA[i]*valB[i];
				//if( tid<32 ) printf("vid:%d, bid:%d, row:%d, col: %d, val:%f\n", i, j, row_idx[i], col_idx[i], valA[i]); 
				if( insert( row_idx[i], col_idx[i], valC[i], d_hashKey, d_hashVal, constants )==false ) atomicAdd( value, 1 );//printf("Error: fail to insert %d, %d, %f\n", row_idx[i], col_idx[i], valC[i]);
			}
	}}
    __syncthreads();

    // Store the values to global memory.
    mgpu::DeviceRegToGlobal<NT, VT>(destCount, gather, tid, output_global + range.x);
}

// Kernel Entry point for performing batch intersection computation
template <typename typeVal>//, typename ProblemData, typename Functor>
    float LaunchKernel( 
		d_matrix *C, d_matrix *A, d_matrix *B, 
        int *d_offA,
        int *d_lengthA,
        int *d_offB,
        int *d_lengthB,
        int *d_interbalance,
        int *d_scanbalance,
		int *h_inter,
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
    cudaMalloc( &d_inter, (partNum*partNum+1)*sizeof(int) );
    cudaMemcpy( d_inter, h_inter, (partNum*partNum+1)*sizeof(int), cudaMemcpyHostToDevice );

	unsigned *d_hashKey;
	float *d_hashVal;
	cudaMalloc( &d_hashKey, TABLE_SIZE*sizeof(unsigned) );
	cudaMalloc( &d_hashVal, TABLE_SIZE*sizeof(float)    );
	//print_array_device( "Hash Keys", d_hashKey, 40 );
	//print_array_device( "Hash Vals", d_hashVal, 40 );

	int *d_value, tempValue, value;
	value = 0;
	cudaMalloc( &d_value, sizeof(int) );
	cudaMemset( d_value, 0, sizeof(int) );

	// Tuning
    const int NT = 128;
    const int VT = 7;
	
	// Constants
	uint2 constants;
	generateConstants( &constants );
	printf("constants: %d %d\n", constants.x, constants.y );

	cudaProfilerStart();
	GpuTimer gpu_timer;
	float elapsed = 0.0f;
	gpu_timer.Start();

	for( int i=0; i<partNum; i++ ) {
		for( int j=0; j<partNum; j++ ) {
	//for( int i=0; i<1; i++ ) {
	//	for( int j=0; j<1; j++ ) {
			cudaMemset( d_hashKey, SLOT_EMPTY_INIT, TABLE_SIZE*sizeof(unsigned) );
			cudaMemset( d_hashVal, 0.0f, TABLE_SIZE*sizeof(float)               );

			int intervalCount = h_inter[partNum*i+j+1]-h_inter[partNum*i+j];
			int moveCount = 0;
			//mgpu::Scan<mgpu::MgpuScanTypeExc>( d_interbalance+h_inter[partNum*i+j], intervalCount, 0, mgpu::plus<int>(), (int*)0, &moveCount, d_scanbalance+h_inter[partNum*i+j], context );
			mgpu::Scan<mgpu::MgpuScanTypeExc>( d_lengthA+h_inter[partNum*i+j], intervalCount, 0, mgpu::plus<int>(), (int*)0, &moveCount, d_scanbalance+h_inter[partNum*i+j], context );

    		typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
    		int2 launch = Tuning::GetLaunchParams(context);

    		int NV = launch.x * launch.y;
    		int numBlocks = MGPU_DIV_UP(moveCount + intervalCount, NV);

    		MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>(
        		mgpu::counting_iterator<int>(0), moveCount, d_scanbalance+h_inter[partNum*i+j],
        		intervalCount, NV, 0, mgpu::less<int>(), context);

    		//int4 range = CTALoadBalance<NT, VT>(moveCount, indices_global, 
        	//	intervalCount, block, tid, mp_global, indices_shared, true);
			//printf("moveCount:%d\n", moveCount);

			KernelInterval<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>( moveCount, d_scanbalance+h_inter[partNum*i+j], d_offA+h_inter[partNum*i+j], d_offB+h_inter[partNum*i+j], d_lengthB+h_inter[partNum*i+j], A->d_dcscRowInd, A->d_dcscVal, B->d_dcscRowInd, B->d_dcscVal, intervalCount, partitionsDevice->get(), d_interbalance, d_hashKey, d_hashVal, d_value, constants );
			//KernelMove<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>( moveCount, d_scanbalance+h_inter[partNum*i+j], intervalCount, mgpu::counting_iterator<int>(0), partitionsDevice->get(), d_interbalance );

			//print_array_device( "good offA", d_interbalance, moveCount );
			//printf("%d %d\n", i, j);
			//print_array_device( "mergepath", partitionsDevice->get(), intervalCount );
			cudaMemcpy( &tempValue, d_value, sizeof(int), cudaMemcpyDeviceToHost );
			value += tempValue;
			//printf("Failed inserts: %d\n", value);

			//Retrieve( d_hashKey, d_hashVal );
		}
	}

	gpu_timer.Stop();
	cudaProfilerStop();
	elapsed += gpu_timer.ElapsedMillis();
	printf("my spgemm: %f ms\n", elapsed);
	printf("Failed inserts: %d\n", value);
	//CudaCheckError();

	print_array("h_inter", h_inter, partNum*partNum+1);
	print_array_device("Off", A->d_dcscColPtr_off+h_inter[1], 40);
	print_array_device("Row", A->d_cscColInd, h_inter[1]);
	print_array_device("Col", A->d_cscRowInd, h_inter[1]);
	print_array_device("Val", A->d_cscVal, h_inter[1]);

}

