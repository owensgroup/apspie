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
#define BLOCK_SIZE 1024
#define TABLE_SIZE 131071
#define MAX_PROBES 10
#define SLOT_EMPTY 0xffffffff
#define SLOT_EMPTY_INIT 255
#define JUMP_HASH 41
//#define PRIME_DIV 33214459
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

//__constant__ unsigned constant1x = 

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

__device__ int BinarySearchStart(int* keys, int count, int key, int *val) {
    int begin = 0;
    int end = count;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        int item = keys[mid];
        if (item == key) { *val=keys[mid]; return mid; }
        bool larger = (item > key);
        if (larger) end = mid;
        else begin = mid+1;
    }
	*val = keys[end-1];
    return end-1;
}

__device__ int BinarySearchEnd(int* keys, int count, int key, int *val) {
    int begin = 0;
    int end = count;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        int item = keys[mid];
        if (item == key) { begin = mid+1; *val=keys[mid]; }
        bool larger = (item > key);
        if (larger) end = mid;
        else begin = mid+1;
    }
	*val = keys[end-1];
    return end-1;
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
		//printf("row:%d, col:%d, val:%f, key:%d, hash:%d, table:%u\n", row_idx, col_idx, valC, key, hash, d_hashKey[hash]);
		//if( d_hashKey[hash] == key ) {
			//atomicAdd( d_hashVal+hash, valC );
			//printf("row:%d, col:%d, val:%f, key:%d, hash:%d, table:%u\n", row_idx, col_idx, valC, key, hash, d_hashKey[hash]);
			//return true;
		//}
		//if( d_hashKey[hash] == SLOT_EMPTY ) {
			unsigned old = atomicCAS( d_hashKey+hash, SLOT_EMPTY, key );
			if( old==SLOT_EMPTY || old==key ) {
				//printf("row:%d, col:%d, val:%f, key:%d, hash:%d, table:%u\n", row_idx, col_idx, valC, key, hash, d_hashKey[hash]);
				atomicAdd( d_hashVal+hash, valC );
				return true;
			}
		//}
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
__global__ void KernelInterval( const int* d_moveCount, 
	IndicesIt indices_global, ValuesIt values_globalA, ValuesIt d_offB, 
	ValuesIt d_lengthB, IndicesIt d_dcscRowIndA, const float *d_dcscValA, 
	IndicesIt d_dcscRowIndB, const float *d_dcscValB, const int *d_inter, 
	const int* d_partitions, OutputIt output_global, unsigned *d_hashKey, 
	float *d_hashVal, int *value, uint2 constants, int* d_blocksBegin,
	int* d_partitionsBegin, const int partNum ) {

    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    typedef typename std::iterator_traits<ValuesIt>::value_type T;

    union Shared {
        int indices[NT * (VT + 1)];
        T values[NT * VT];
    };
    __shared__ Shared shared;
	__shared__ int idx;
	__shared__ int idxEnd;
	__shared__ int moveCount;       // d_moveCount[idx]
	__shared__ int intervalCount;   // d_inter[idx+1] - d_inter[idx]
	__shared__ int inter;           // d_inter[idx]
	__shared__ int blocksBegin;     // d_blocksBegin[idx]
	__shared__ int partitionsBegin; // d_partitionsBegin[idx]

    int tid = threadIdx.x;
    int block = blockIdx.x;

    if( tid==0 ) {
		//idx = BinarySearchStart( d_blocksBegin, partNum*partNum+1, block, 
		//	&blocksBegin );
		idx = BinarySearchEnd( d_blocksBegin, partNum*partNum+1, block, 
			&blocksBegin );
		moveCount = __ldg( d_moveCount + idx );
		inter = __ldg( d_inter + idx );
		intervalCount = __ldg( d_inter + idx + 1 ) - inter;
		partitionsBegin = __ldg( d_partitionsBegin + idx );
		//if( block%100==0 ) 
		//	printf("block:%d,idx:%d,mov:%d,int:%d,intC:%d,par:%d\n", block, idx, moveCount, inter, intervalCount, blocksBegin);
	}

	__syncthreads();

	//int row_i = idx/partNum;
	//int col_j = idx%partNum;
	int target = idx*TABLE_SIZE;
	//if( tid==0 ) printf("target:%d\n", target);
	
    // Compute the input and output intervals this CTA processes.
    int4 range = mgpu::CTALoadBalance<NT, VT>( moveCount, indices_global+inter, 
		intervalCount, block-blocksBegin, tid, d_partitions+partitionsBegin,
		shared.indices, true);

    // The interval indices are in the left part of shared memory (moveCount).
    // The scan of interval counts are in the right part (intervalCount).
    int destCount = range.y - range.x;
    int sourceCount = range.w - range.z;
	//if( tid==0 && block>1050 && block<1100 ) printf("block:%d, tid:%d, idx:%d, %d, %d, %d, %d\n", block, tid, idx, range.x, range.y, range.z, range.w, destCount, sourceCount );

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
    mgpu::DeviceMemToMemLoop<NT>(sourceCount, values_globalA + inter + range.z, 		tid, shared.values);

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
			length[i] = d_lengthB[inter+sources[i]];    // lengthB
			off[i]    = d_offB[inter+sources[i]];        // offB
			row_idx[i]= d_dcscRowIndA[gather[i]];
			valA[i]   = d_dcscValA[gather[i]];
			/*length[i] = __ldg(d_lengthB+inter+sources[i]);    // lengthB
			off[i]   = __ldg(d_offB+inter+sources[i]);        // offB
			row_idx[i]= __ldg(d_dcscRowIndA+gather[i]);
			valA[i]   = __ldg(d_dcscValA+gather[i]);*/
			//if( tid<32 ) printf("tid:%d, vid:%d, gather: %d, off:%d, length:%d, row:%d, val:%f, inter:%d\n", tid, i, gather[i], off[i], length[i], row_idx[i], valA[i], inter); 

			for( int j=0; j<length[i]; j++ ) {
				col_idx[i] = d_dcscRowIndB[off[i]+j];
				valB[i]    = d_dcscValB[off[i]+j];
				//col_idx[i] = __ldg(d_dcscRowIndB+off[i]+j);
				//valB[i]    = __ldg(d_dcscValB+off[i]+j);
				valC[i]    = valA[i]*valB[i];
				//if( tid<32 ) printf("vid:%d, bid:%d, row:%d, col: %d, val:%f, idx:%d\n", i, j, row_idx[i], col_idx[i], valC[i], idx); 
				if( insert( row_idx[i], col_idx[i], valC[i], d_hashKey+target, d_hashVal+target, constants )==false ) atomicAdd( value, 1 );//printf("Error: fail to insert %d, %d, %f\n", row_idx[i], col_idx[i], valC[i]);
			}
	}}
    //__syncthreads();

    // Store the values to global memory.
    //mgpu::DeviceRegToGlobal<NT, VT>(destCount, gather, tid, output_global + range.x);
}

template<typename Tuning, typename IndicesIt, typename ValuesIt, typename OutputIt>
__global__ void KernelMove(int destCount,
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
    int4 range = mgpu::CTALoadBalance<NT, VT>(destCount, indices_global, 
		sourceCount, block, tid, mp_global, shared.indices, true);

    // The interval indices are in the left part of shared memory (moveCount).
    // The scan of interval counts are in the right part (intervalCount).
    destCount = range.y - range.x;
    sourceCount = range.w - range.z;
	//if( tid==0 && block<50 ) printf("block:%d, tid:%d, %d, %d, %d, %d\n", block, tid, range.x, range.y, range.z, range.w, destCount, sourceCount );

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
    //__syncthreads();

    // Store the values to global memory.
    //mgpu::DeviceRegToGlobal<NT, VT>(destCount, gather, tid, output_global + range.x);
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
		const int *d_moveCount,
		const int *h_moveCount,
		const int partSize,
		const int partNum,
		const int moveTotal,
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
    CUDA_SAFE_CALL(cudaMalloc( &d_inter, (partNum*partNum+1)*sizeof(int) ));
    CUDA_SAFE_CALL(cudaMemcpy( d_inter, h_inter, (partNum*partNum+1)*sizeof(int), cudaMemcpyHostToDevice ));

	unsigned *d_hashKey, *h_hashKey, *d_hashKeyTemp;
	float *d_hashVal, *h_hashVal, *d_hashValTemp;
	CUDA_SAFE_CALL(cudaMalloc( &d_hashKey, partNum*partNum*TABLE_SIZE*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMalloc( &d_hashVal, partNum*partNum*TABLE_SIZE*sizeof(float)    ));
	CUDA_SAFE_CALL(cudaMemset( d_hashKey, SLOT_EMPTY_INIT, partNum*partNum*TABLE_SIZE*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMemset( d_hashVal, 0.0f, partNum*partNum*TABLE_SIZE*sizeof(float)));
	h_hashKey = (unsigned*) malloc( TABLE_SIZE*sizeof(unsigned) );
	h_hashVal = (float*) malloc( TABLE_SIZE*sizeof(float) );
	CUDA_SAFE_CALL(cudaMalloc( &d_hashKeyTemp, TABLE_SIZE*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMalloc( &d_hashValTemp, TABLE_SIZE*sizeof(float)    ));
	//print_array_device( "Hash Keys", d_hashKey, 40 );
	//print_array_device( "Hash Vals", d_hashVal, 40 );

	int *d_value, value;
	value = 0;
	CUDA_SAFE_CALL(cudaMalloc( &d_value, sizeof(int) ));
	CUDA_SAFE_CALL(cudaMemset( d_value, 0, sizeof(int) ));

	// Tuning
    const int NT = 128;
    const int VT = 1;
	
	// Constants
	uint2 constants;
	for( int i=0; i<5; i++ ) {
	generateConstants( &constants );
	printf("__constant__ unsigned constant %u %u\n", constants.x, constants.y );
	generateConstants( &constants );
	printf("constants: %u %u\n", constants.x, constants.y );
	}

	int *h_partitionsBegin, *d_partitionsBegin;
	h_partitionsBegin = (int*) malloc( (partNum*partNum+1)*sizeof(int) );
	h_partitionsBegin[0] = 0;
	CUDA_SAFE_CALL(cudaMalloc( &d_partitionsBegin, (partNum*partNum+1)*sizeof(int) ));
	int *h_blocksBegin, *d_blocksBegin;
	h_blocksBegin = (int*) malloc( (partNum*partNum+1)*sizeof(int) );
	h_blocksBegin[0] = 0;
	CUDA_SAFE_CALL(cudaMalloc( &d_blocksBegin, (partNum*partNum+1)*sizeof(int) ));
	int *d_partitions;
	CUDA_SAFE_CALL(cudaMalloc( &d_partitions, moveTotal*sizeof(int) ));

	typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	// Temp Arrays for Testing
	unsigned *d_hashKeyTest, *h_hashKeyTest;
	float *d_hashValTest, *h_hashValTest;
	CUDA_SAFE_CALL(cudaMalloc( &d_hashKeyTest, TABLE_SIZE*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMalloc( &d_hashValTest, TABLE_SIZE*sizeof(float)    ));
	CUDA_SAFE_CALL(cudaMemset( d_hashKeyTest, SLOT_EMPTY_INIT, TABLE_SIZE*sizeof(unsigned) ));
	CUDA_SAFE_CALL(cudaMemset( d_hashValTest, 0.0f, TABLE_SIZE*sizeof(float)));
	h_hashKeyTest = (unsigned*) malloc( TABLE_SIZE*sizeof(unsigned) );
	h_hashValTest = (float*) malloc( TABLE_SIZE*sizeof(float) );
	for( int i=0; i<1; i++ ) {
	//for( int i=1; i<2; i++ ) {
		for( int j=0; j<1; j++ ) {
			int intervalCount = h_inter[partNum*i+j+1]-h_inter[partNum*i+j];
			int moveCount = h_moveCount[partNum*i+j];
    		int NV = launch.x * launch.y;
    		int numBlocks = MGPU_DIV_UP(moveCount + intervalCount, NV);
			if( moveCount!=0 ) {
    			MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions
					<mgpu::MgpuBoundsUpper>( mgpu::counting_iterator<int>(0), 
					moveCount, d_scanbalance+h_inter[partNum*i+j], 
					intervalCount, NV, 0, mgpu::less<int>(), context);
				KernelMove<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>( moveCount, d_scanbalance+h_inter[partNum*i+j], d_offA+h_inter[partNum*i+j], d_offB+h_inter[partNum*i+j], d_lengthB+h_inter[partNum*i+j], A->d_dcscRowInd, A->d_dcscVal, B->d_dcscRowInd, B->d_dcscVal, intervalCount, partitionsDevice->get(), d_interbalance, d_hashKeyTest, d_hashValTest, d_value, constants );
			}
		}
	}
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d_hashKeyTest, d_hashKeyTemp, d_hashValTest, d_hashValTemp, TABLE_SIZE );
	CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d_hashKeyTest, d_hashKeyTemp, d_hashValTest, d_hashValTemp, TABLE_SIZE );
	CudaCheckError();
	//print_array_device("hashKey multikernel", d_hashKeyTemp, 40 );
	//print_array_device("hashVal multikernel", d_hashValTemp, 40 );
	cudaMemcpy( h_hashKeyTest, d_hashKeyTemp, TABLE_SIZE*sizeof(unsigned), 
		cudaMemcpyDeviceToHost );
	cudaMemcpy( h_hashValTest, d_hashValTemp, TABLE_SIZE*sizeof(float), 
		cudaMemcpyDeviceToHost );
	cudaMemcpy( &value, d_value, sizeof(int), cudaMemcpyDeviceToHost );
	printf("Failed inserts: %d\n", value);
	printf("uni-kernel:\n");

	cudaProfilerStart();
	GpuTimer gpu_timer;
	float elapsed = 0.0f;
	gpu_timer.Start();

	for( int i=0; i<partNum; i++ ) {
		for( int j=0; j<partNum; j++ ) {

			int intervalCount = h_inter[partNum*i+j+1]-h_inter[partNum*i+j];
			int moveCount = h_moveCount[partNum*i+j];
			//mgpu::Scan<mgpu::MgpuScanTypeExc>( d_lengthA+h_inter[partNum*i+j], intervalCount, 0, mgpu::plus<int>(), (int*)0, &moveCount, d_scanbalance+h_inter[partNum*i+j], context );

			//if( moveCount!=h_moveCount[partNum*i+j] ) 
			//	printf("Error: %d!=%d\n", moveCount, h_moveCount[partNum*i+j] );

    		int NV = launch.x * launch.y;
    		int numBlocks = MGPU_DIV_UP(moveCount + intervalCount, NV);

			if( moveCount!=0 ) {

    			MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions
					<mgpu::MgpuBoundsUpper>( mgpu::counting_iterator<int>(0), 
					moveCount, d_scanbalance+h_inter[partNum*i+j], 
					intervalCount, NV, 0, mgpu::less<int>(), context);

    		//int4 range = CTALoadBalance<NT, VT>(moveCount, indices_global, 
        	//	intervalCount, block, tid, mp_global, indices_shared, true);
			//printf("moveCount:%d\n", moveCount);

				h_blocksBegin[partNum*i+j+1] = h_blocksBegin[partNum*i+j]+numBlocks;
				h_partitionsBegin[partNum*i+j+1] = 
					h_partitionsBegin[partNum*i+j]+partitionsDevice->Size();

				//print_array_device( "good offA", d_interbalance, moveCount );
				//printf("%d %d: %d\n", i, j, partitionsDevice->Size());
				//print_array_device( "mergepath", partitionsDevice->get(), 
				//	intervalCount );

				//Retrieve( d_hashKey, d_hashVal );

				cudaMemcpy( d_partitions+h_partitionsBegin[partNum*i+j], 
					partitionsDevice->get(), partitionsDevice->Size()*
					sizeof(int), cudaMemcpyDeviceToDevice );
			} else {
				h_blocksBegin[partNum*i+j+1] = h_blocksBegin[partNum*i+j];
				h_partitionsBegin[partNum*i+j+1] = 
					h_partitionsBegin[partNum*i+j];
			}
		}
	}
	cudaMemcpy( d_blocksBegin, h_blocksBegin, (partNum*partNum+1)*sizeof(int), 
		cudaMemcpyHostToDevice );
	cudaMemcpy( d_partitionsBegin, h_partitionsBegin, (partNum*partNum+1)*
		sizeof(int), cudaMemcpyHostToDevice );
	//verify( partNum*partNum+1, h_blocksBegin, h_partitionsBegin );

	KernelInterval<Tuning><<<h_blocksBegin[partNum*partNum], launch.x, 0, 
		context.Stream()>>>( d_moveCount, d_scanbalance, d_offA, d_offB, 
		d_lengthB, A->d_dcscRowInd, A->d_dcscVal, B->d_dcscRowInd, B->d_dcscVal,
		d_inter, d_partitions, d_interbalance, d_hashKey, d_hashVal, 
		d_value, constants, d_blocksBegin, d_partitionsBegin, partNum );

	gpu_timer.Stop();
	cudaProfilerStop();
	elapsed += gpu_timer.ElapsedMillis();
	printf("my spgemm: %f ms\n", elapsed);
	cudaMemcpy( &value, d_value, sizeof(int), cudaMemcpyDeviceToHost );
	printf("Failed inserts: %d\n", value);
	//CudaCheckError();

	print_array_device( "d_blocksBegin", d_blocksBegin, partNum*partNum+1 );
	print_array_device( "d_partitionsBegin", d_partitionsBegin, partNum*
		partNum+1 );
	printf("Blocks launched: %d\n", h_blocksBegin[partNum*partNum]);
	


	// Radix sort
	GpuTimer gpu_timer2;
	float elapsed2 = 0.0f;
	gpu_timer2.Start();

	CUDA_SAFE_CALL(cudaFree(d_temp_storage));
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d_hashKey, d_hashKeyTemp, d_hashVal, d_hashValTemp, TABLE_SIZE );
	CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, d_hashKey, d_hashKeyTemp, d_hashVal, d_hashValTemp, TABLE_SIZE );

	gpu_timer2.Stop();
	elapsed2 += gpu_timer2.ElapsedMillis();
	printf("radix sort: %f ms\n", elapsed2);
	CudaCheckError();

	print_array_device("hashKey", d_hashKeyTemp, 40 );
	print_array_device("hashVal", d_hashValTemp, 40 );

	/*cudaMemcpy( h_hashKeyTest, d_hashKey, TABLE_SIZE*sizeof(unsigned),
		cudaMemcpyDeviceToHost );
	cudaMemcpy( h_hashValTest, d_hashVal, TABLE_SIZE*sizeof(float),
		cudaMemcpyDeviceToHost );*/
	cudaMemcpy( h_hashKey, d_hashKeyTemp, TABLE_SIZE*sizeof(unsigned),
		cudaMemcpyDeviceToHost );
	cudaMemcpy( h_hashVal, d_hashValTemp, TABLE_SIZE*sizeof(float),
		cudaMemcpyDeviceToHost );
	verify( TABLE_SIZE, h_hashKey, h_hashKeyTest );
	verify( TABLE_SIZE, h_hashVal, h_hashValTest );

	print_array_device("d_interbalance", d_interbalance, 40);
	print_array_device("Row", A->d_cscColInd, h_inter[1]);
	print_array_device("Col", A->d_cscRowInd, h_inter[1]);
	print_array_device("Val", A->d_cscVal, h_inter[1]);
}

