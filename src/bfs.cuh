// Provides BFS function for GPU

#include <cuda_profiler_api.h>
#include <cusparse.h>
#include "mXv.cuh"
#include "scratch.hpp"

//#define NBLOCKS 16384
#define NTHREADS 512

template<typename T>
void spmv( const T *d_inputVector, const int new_nnz, const int m, const T *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, T *d_spmvResult, mgpu::CudaContext& context) {
    mgpu::SpmvCsrBinary(d_csrValA, d_csrColIndA, new_nnz, d_csrRowPtrA, m, d_inputVector, true, d_spmvResult, (T)0, mgpu::multiplies<T>(), mgpu::plus<T>(), context);
}

__global__ void bitifySparse( const int *d_randVec, int *d_randVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x ) {
        if( d_randVec[idx]>-1 ) d_randVecInd[idx] = 1;
        else d_randVecInd[idx] = 0;
    }
}

// @brief Performs global to local format conversion before updating BFSResult
//
__global__ void addResultSparse( int *d_bfsResult, int* d_spmvSwapInd, const int iter, const int length, const int rank, const int h_size ) {
    const int STRIDE = gridDim.x * blockDim.x;
	const int offset = rank*h_size;
    for( int idx = (blockIdx.x * blockDim.x ) + threadIdx.x; idx<length; idx+=STRIDE ) {
        if( d_spmvSwapInd[idx]>=0 && d_bfsResult[d_spmvSwapInd[idx]-offset] < 0 ) {
            d_bfsResult[d_spmvSwapInd[idx]-offset] = iter;
			d_spmvSwapInd[idx] -= offset;
        } else d_spmvSwapInd[idx] = -1;
    }
}

template< typename T >
__global__ void streamCompactSparse( const T *d_tempVal, const int *d_cscFlag, const int *d_cscColGood, T *d_cscVecInd, const int m ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<m; idx+=blockDim.x*gridDim.x )
        if( d_cscFlag[idx] ) d_cscVecInd[d_cscColGood[idx]]=d_tempVal[idx];
}

__device__ void swap(int &first, int &second) {
    int temp;
    temp = first;
    first = second;
    second = temp;
    return;
}

__global__ void generateHistogram( const int new_n, const int nnz, const int *d_spmvSwapInd, int *d_sendHist, int *d_mutex ) {
    for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<nnz-1; idx+=blockDim.x*gridDim.x ) {

    // This code is checking whether we have come to a boundary in GPU partition dest.
    // Should this be a <?
        if( d_spmvSwapInd[idx]/new_n < d_spmvSwapInd[idx+1]/new_n ) {
            //printf("<=\n");
            bool isSet = false;
            do {
                if( isSet = atomicCAS( d_mutex, 0, 1 ) == 0 ) {
                    d_sendHist[d_spmvSwapInd[idx+1]/new_n] = idx+1;
                } if( isSet ) {
                    *d_mutex = 0;
                }
            } while( !isSet );
        }
    }
}

// @brief Generates the bin (GPU#) that each SPMV result falls into
//
__global__ void generateKey( const int new_n, const int nnz, const int *d_spmvSwapInd, int *d_cscFlag ) 
{
	for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<nnz; idx+=blockDim.x*gridDim.x ) 
		d_cscFlag[idx] = d_spmvSwapInd[idx]/new_n;
}

// @brief Performs global to local conversion on cscColPtr
//
__global__ void updateColPtr( int *d_cscColPtr, const int length ) {
	const int offset = d_cscColPtr[0];
	for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<length; idx+=blockDim.x*gridDim.x ) {
		d_cscColPtr[idx] -= offset;
	}
}

// @brief Makes right element -1 if it is equal to the current element
//
__global__ void lookRightUnique( int *d_array, const int length ) {
	for( int idx=blockDim.x*blockIdx.x+threadIdx.x; idx<length-1; idx+=blockDim.x*gridDim.x ) {
		if( d_array[idx] == d_array[idx+1] ) d_array[idx] = -1;
	}
}

template< typename T >
void bfsSparse( const int vertex, const int new_nnz, const int new_n, const int old_n, const int multi, const int rank, const T* d_cscValA, int *d_cscColPtrA, const int *d_cscRowIndA, int *d_bfsResult, const int depth, mgpu::CudaContext& context) {

    /*cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);*/

    // Allocate scratch memory
    d_scratch *d;
    allocScratch( &d, new_nnz, old_n );

    // Allocate GPU memory for result
    int *h_bfsResult = (int*)malloc(old_n*sizeof(int));

    int *h_spmvResultInd = (int*)malloc(old_n*sizeof(int));
    int *h_spmvSwapInd = (int*)malloc(old_n*sizeof(int));
    float *h_spmvResultVec = (float*)malloc(old_n*sizeof(float));
    float *h_spmvSwapVec = (float*)malloc(old_n*sizeof(float));

    int *d_spmvResultInd, *d_spmvSwapInd;
    float *d_spmvResultVec, *d_spmvSwapVec;
    cudaMalloc(&d_spmvResultInd, old_n*sizeof(int));
    cudaMalloc(&d_spmvSwapInd, old_n*sizeof(int));
    cudaMalloc(&d_spmvResultVec, old_n*sizeof(float));
    cudaMalloc(&d_spmvSwapVec, old_n*sizeof(float));
    
    int h_nnz = 0;
    int h_cscVecCount = 0;

    // Allocate histogram of send 
    int *h_sendHist = (int*)malloc(multi*sizeof(int));
    //int *h_sendHistProc = (int*)malloc(multi*sizeof(int));
    int *h_recvHist = (int*)malloc(multi*sizeof(int));

    int *d_sendHist, *d_recvHist;
    cudaMalloc(&d_sendHist, multi*sizeof(int));
    cudaMalloc(&d_recvHist, multi*sizeof(int));

    // Allocate prefix sum of send 
    int *h_sendScan = (int*)malloc((multi+1)*sizeof(int));
    int *h_recvScan = (int*)malloc((multi+1)*sizeof(int));

    int *d_sendScan, *d_recvScan;
    cudaMalloc(&d_sendScan, (multi+1)*sizeof(int));
    cudaMalloc(&d_recvScan, (multi+1)*sizeof(int));

    // Only make one element of h_bfsResult 0 if following is true:
    //   a) if not last processor
    //   b) if last processor
    // Also generate initial sparse vector using vertex
    for( int i=0; i<new_n; i++ ) {
        h_bfsResult[i] = -1;
        if( ((rank != multi-1 && rank == vertex/new_n) || (rank==multi-1 && vertex >= rank*(old_n+multi-1)/multi) ) && i==vertex ) {
            h_bfsResult[vertex-rank*(old_n/multi+1)] = 0;
            h_spmvResultInd[0] = vertex-rank*(old_n/multi+1);
			h_spmvResultVec[0] = 1.0;
			h_nnz = 1;
            //printf("Source vertex on processor %d!\n", rank);
        }
    }

    // Generate d_ones, d_index
    for( int i=0; i<old_n; i++ ) {
        d->h_ones[i] = 1;
        d->h_index[i] = i;
    }
	//print_array(d->h_spmvResult, 40);
    cudaMemcpy(d_bfsResult, h_bfsResult, new_n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultInd, h_spmvResultInd, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spmvResultVec, h_spmvResultVec, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_index, d->h_index, old_n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d->d_ones, d->h_ones, old_n*sizeof(int), cudaMemcpyHostToDevice);

    // Allocate counter and mutex (for histogram)
    int h_counter = 0;
    int *d_mutex;
    cudaMalloc(&d_mutex, sizeof(int));
    cudaMemcpy(d_mutex, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Generate d_cscColDiff
    int NBLOCKS = (new_n+NTHREADS-1)/NTHREADS;
    diff<<<NBLOCKS,NTHREADS>>>(d_cscColPtrA, d->d_cscColDiff, new_n);

    // Global to local cscColPtr conversion
	//printf("%d: PreLocal:\n", rank);
	//print_device( d_cscColPtrA, new_n+1 );
	//if( rank==3 ) printDevice( "ColPtr", d_cscColPtrA );
	updateColPtr<<<NBLOCKS,NTHREADS>>>( d_cscColPtrA, new_n+1 );
	//if( rank==0 || rank==3 ) printDevice( "ColPtr", d_cscColPtrA );
	//if( rank==0 || rank==3 ) printDevice( "RowInd", d_cscRowIndA );
	//printDevice( "PostLocal", d_cscColPtrA, new_n+1 );

	// Testing correctness:
	printf("Rank %d, new_nnz %d, new_n %d, old_n %d, multi %d\n", rank, new_nnz, new_n, old_n, multi);
    int h_size = (old_n+multi-1)/multi;

	// File output:
	char filename[20];
    sprintf(filename, "file_%d.out", rank);
    std::ofstream outf(filename);

    // Keep a count of new_nnzs traversed
	int h_send;
    int cumsum = 0;
    int sum = 0;
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cudaProfilerStart();

	// Important that i begins at 1, because it is used to update BFS result
    //for( int i=1; i<3; i++ ) {
    for( int i=1; i<depth; i++ ) {
			outf << "Iteration " << i << ": " << rank << std::endl << "==========";
            //spmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, context);
            //cuspmv<float>( d_spmvResult, new_nnz, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwap, handle, descr);

            // op=1 Arithmetic semiring
            //sum = mXvSparse( d_spmvResultInd, d_spmvResultVec, new_nnz, new_n, old_n, h_nnz, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwapInd, d_spmvSwapVec, d, context);
            sum = mXvSparseDebug( d_spmvResultInd, d_spmvResultVec, new_nnz, new_n, old_n, h_nnz, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_spmvSwapInd, d_spmvSwapVec, d, outf, context);
			//fprintDevice("mXvSparse", outf, d_spmvSwapInd, h_nnz);

            // Generate the send prefix sums
			
			// Option 1: Histogram
            //generateHistogram<<<NTHREADS, NBLOCKS>>>( h_size, h_nnz, d_spmvSwapInd, d_sendScan, d_mutex );
			//
			// Max out sendScan from last index of SpmvResult
            //cudaMemcpy( h_sendScan, d_sendScan, (multi+1)*sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy( &h_counter, &d_spmvSwapInd[h_nnz-1], sizeof(int), cudaMemcpyDeviceToHost);
			//for( int j=h_counter/h_size+1; j<multi+1; j++ )
			//	h_sendScan[j] = h_nnz;
			//
			// Zero out sendScan from previous iterations
			//if( h_nnz==0 )
			//	for( int j=0; j<multi+1; j++ )
			//		h_sendScan[j] = 0;
			//fprintArray("SendScan", outf, h_sendScan, multi+1);
			//
			// Linear undo prefix sum using CPU
			//linearUnscan( h_sendScan, h_sendHist, multi );
			//fprintArray("SendHist", outf, h_sendHist, multi);

			//
			// Option 2: Generate Bin ID in parallel and Segmented Reduce
			//
			// Zero out sendScan from previous iterations
			if( h_nnz==0 )
				zeroArray<<<NTHREADS,NBLOCKS>>>( d_sendHist, multi );
			else {
				zeroArray<<<NTHREADS,NBLOCKS>>>( d->d_cscColBad, h_send );
				zeroArray<<<NTHREADS,NBLOCKS>>>( d_sendHist, multi );
				generateKey<<<NTHREADS,NBLOCKS>>>( h_size, h_nnz, d_spmvSwapInd, d->d_cscColGood );
				//outf << "h_size: \n" << h_size << std::endl;
				//fprintDevice("Generate Key", outf, d->d_cscColGood, h_nnz);
				//fprintDevice("Array of 1's", outf, d->d_ones, h_nnz);

				//outf.flush();
				ReduceByKey( d->d_cscColGood, d->d_ones, h_nnz, (int)0, mgpu::plus<int>(), mgpu::equal_to<int>(), d->d_cscColBad, d->d_cscVecInd, &h_send, (int*)0, context );
				//fprintDevice("ReduceByKey Key", outf, d->d_cscColBad, h_send);
				//fprintDevice("ReduceByKey Val", outf, d->d_cscVecInd, h_send);

				//outf.flush();
				scatterFloat<<<NTHREADS,NBLOCKS>>>( h_send, d->d_cscColBad, d->d_cscVecInd, d_sendHist );
			}
			//fprintDevice("SendHist", outf, d_sendHist, multi);

			cudaMemcpy( h_sendHist, d_sendHist, multi*sizeof(int), cudaMemcpyDeviceToHost );
			linearScan( h_sendHist, h_sendScan, multi);
			//fprintArray("SendScan", outf, h_sendScan, multi+1);

            // Exchange send prefix sums
			MPI_Barrier( MPI_COMM_WORLD );
            MPI_Alltoall( h_sendHist, 1, MPI_INT, h_recvHist, 1, MPI_INT, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			//fprintArray("RecvHist", outf, h_recvHist, multi);

			// Linear prefix sum using CPU
			linearScan( h_recvHist, h_recvScan, multi );
			//fprintArray("Pre-Alltoallv RecvScan", outf, h_recvScan, multi+1);

            // Exchange vectors
			outf.flush();
			MPI_Barrier( MPI_COMM_WORLD );
            MPI_Alltoallv( d_spmvSwapInd, h_sendHist, h_sendScan, MPI_INT, d_spmvResultInd, h_recvHist, h_recvScan, MPI_INT, MPI_COMM_WORLD );
            MPI_Alltoallv( d_spmvSwapVec, h_sendHist, h_sendScan, MPI_INT, d_spmvResultVec, h_recvHist, h_recvScan, MPI_INT, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			//fprintDevice("Pre-sort Frontier", outf, d_spmvResultInd, h_recvScan[multi]);

            // Merge vectors
			// 2 options:
			//	1) merge sparse vectors
			//  2) atomicAdd into dense vector
			if( h_recvScan[multi] != 0 ) { 
            	MergesortPairs( d_spmvResultInd, d_spmvResultVec, h_recvScan[multi], mgpu::less<int>(), context );
				lookRightUnique<<<NBLOCKS,NTHREADS>>>( d_spmvResultInd, h_recvScan[multi] );
			}
			//fprintDevice("SortPairs", outf, d_spmvResultInd, h_recvScan[multi] );

            // Update BFS Result
            addResultSparse<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResultInd, i, h_recvScan[multi], rank, h_size );
			//fprintDevice("BFSResult", outf, d_bfsResult, h_size);
			//fprintDevice("Pre-Filter SpmvResultInd", outf, d_spmvResultInd, h_recvScan[multi]);

            // Prune new vector
			cudaMemcpy(d_spmvSwapInd, d_spmvResultInd, h_recvScan[multi]*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_spmvSwapVec, d_spmvResultVec, h_recvScan[multi]*sizeof(float), cudaMemcpyDeviceToDevice);
            bitifySparse<<<NBLOCKS,NTHREADS>>>( d_spmvResultInd, d->d_randVecInd, h_recvScan[multi] );
			//fprintDevice("Bitify Sparse", outf, d->d_randVecInd, h_recvScan[multi]);
            mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_randVecInd, h_recvScan[multi], 0, mgpu::plus<int>(), (int*)0, &h_cscVecCount, d->d_cscColGood, context );
			//fprintDevice("Indices Good", outf, d->d_cscColGood, h_recvScan[multi]);
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapInd, d->d_randVecInd, d->d_cscColGood, d_spmvResultInd, h_recvScan[multi] );
            streamCompactSparse<<<NBLOCKS,NTHREADS>>>( d_spmvSwapVec, d->d_randVecInd, d->d_cscColGood, d_spmvResultVec, h_recvScan[multi] );           
            h_nnz = h_cscVecCount;
			//fprintDevice("Post-Filter SpmvResultInd", outf, d_spmvResultInd, h_nnz);

        cumsum+=sum;
    }

    cudaProfilerStop();
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("\nGPU%d BFS finished in %f msec. \n", rank, elapsed);
    printf("Traversed new_nnzs: %d\n", cumsum);
    printf("Performance: %f GTEPS\n", (float)cumsum/(elapsed*1000000));
	
    // Important: destroy handle
    //cusparseDestroy(handle);
    //cusparseDestroyMatDescr(descr);
}
