// Puts everything together
// For now, just run V times.
// Optimizations: 
// -come up with good stopping criteria [done]
// -start from i=1 [done]
// -test whether float really are faster than ints
// -distributed idea
// -

#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusparse.h>

#include <cublas_v2.h>
#include <sys/resource.h>
#include <deque>

#define N (100*1024)
//#define NBLOCKS 16384
#define NTHREADS 1024
#define MARK_PREDECESSORS 0

template<typename T>
void print_end_interesting( T *array, int length ) {
    int count=0;
    for( int j=length-1;j>=0; j-- ) {
        if( array[(int)j]!=-1) {
            std::cout << "[" << j << "]:" << array[j] << " ";
            count++;
            if( count==9 ) break;
        }
    }
    std::cout << "\n";
}

template<typename T>
void print_end( T *array, int length ) {
    int start = length > 10 ? length-10 : 0;
    for( int j=start;j<length;j++ ) {
        std::cout << array[j] << " ";
    }
    std::cout << "\n";
}

template<typename T>
void print_array( T *array, int length ) {
    if( length>40 ) length=40;
    for( int j=0;j<length;j++ ) {
        std::cout << "[" << j << "]:" << array[j] << " ";
    }
    std::cout << "\n";
}

struct CpuTimer {
    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec /1000);
    }

};

/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

 /**
  * @brief A simple CPU-based reference BFS ranking implementation.
  *
  * @tparam VertexId
  * @tparam Value
  * @tparam SizeT
  *
  * @param[in] graph Reference to the CSR graph we process on
  * @param[in] source_path Host-side vector to store CPU computed labels for each node
  * @param[in] predecessor Host-side vector to store CPU computed predecessor for each node
  * @param[in] src Source node where BFS starts
  */
template<typename VertexId>
int SimpleReferenceBfs(
    const VertexId m, const VertexId *h_rowPtrA, const VertexId *h_colIndA,
    VertexId                                *source_path,
    VertexId                                *predecessor,
    VertexId                                src,
    VertexId                                stop)
{
    //initialize distances
    for (VertexId i = 0; i < m; ++i) {
        source_path[i] = -1;
        if (MARK_PREDECESSORS)
            predecessor[i] = -1;
    }
    source_path[src] = 0;
    VertexId search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::deque<VertexId> frontier;
    frontier.push_back(src);

    //
    //Perform BFS
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty()) {
        
        // Dequeue node from frontier
        VertexId dequeued_node = frontier.front();
        frontier.pop_front();
        VertexId neighbor_dist = source_path[dequeued_node] + 1;
        if( neighbor_dist > stop )
            break;

        // Locate adjacency list
        int edges_begin = h_rowPtrA[dequeued_node];
        int edges_end = h_rowPtrA[dequeued_node + 1];

        for (int edge = edges_begin; edge < edges_end; ++edge) {
            //Lookup neighbor and enqueue if undiscovered
            VertexId neighbor = h_colIndA[edge];
            if (source_path[neighbor] == -1) {
                source_path[neighbor] = neighbor_dist;
                if (MARK_PREDECESSORS) {
                    predecessor[neighbor] = dequeued_node;
                }
                if (search_depth < neighbor_dist) {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            }
        }
    }

    if (MARK_PREDECESSORS)
        predecessor[src] = -1;

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    printf("CPU BFS finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

    return search_depth;
}

int bfsCPU( const int src, const int m, const int *h_rowPtrA, const int *h_colIndA, int *h_bfsResultCPU, const int stop ) {

    typedef int VertexId; // Use as the node identifier type

    VertexId *reference_check_preds = NULL;

    int depth = SimpleReferenceBfs<VertexId>(
        m, h_rowPtrA, h_colIndA,
        h_bfsResultCPU,
        reference_check_preds,
        src,
        stop);

    //print_array(h_bfsResultCPU, m);
    return depth;
}
/******************************************************************************
 * Helper routines for list construction and validation 
 ******************************************************************************/

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Compares the equivalence of two arrays. If incorrect, print the location
 * of the first incorrect value appears, the incorrect value, and the reference
 * value.
 * \return Zero if two vectors are exactly the same, non-zero if there is any difference.
 *
 */
template <typename T>
int CompareResults(T* computed, T* reference, int len, bool verbose = true)
{
    int flag = 0;
    for (int i = 0; i < len; i++) {

        if (computed[i] != reference[i] && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            std::cout << computed[i];
            printf(" != ");
            std::cout << reference[i];

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << computed[j];
                    printf(", ");
                }
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << reference[j];
                    printf(", ");
                }
                printf("...]");
            }
            flag += 1;
            //return flag;
        }
        if (computed[i] != reference[i] && flag > 0) flag+=1;
    }
    printf("\n");
    if (flag == 0)
        printf("CORRECT\n");
    return flag;
}

// Verify the result
void verify( const int m, const int *h_bfsResult, const int *h_bfsResultCPU ){
    if (h_bfsResultCPU != NULL) {
        printf("Label Validity: ");
        int error_num = CompareResults(h_bfsResult, h_bfsResultCPU, m, true);
        if (error_num > 0) {
            printf("%d errors occurred.\n", error_num);
        }
    }
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


void coo2csr( const int *d_cooRowIndA, const int edge, const int m, int *d_csrRowPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("COO -> CSR conversion successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid value for idxbase.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
}

void csr2csc( const int m, const int edge, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_cscValA, int *d_cscRowIndA, int *d_cscColPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // For CUDA 4.0
    //cusparseStatus_t status = cusparseScsr2csc(handle, m, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, 1, CUSPARSE_INDEX_BASE_ZERO);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsr2csc(handle, m, m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("Transpose successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
}

void axpy( float *d_spmvSwap, const float *d_csrValA, const int m ) {
    const float alf = -1;
    const float *alpha = &alf;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasSaxpy(handle, 
                            m, 
                            alpha, 
                            d_csrValA, 
                            1, 
                            d_spmvSwap, 
                            1);

    switch( status ) {
        case CUBLAS_STATUS_SUCCESS:
	    //printf("axpy completed successfully!\n");
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
	    printf("The library was not initialized.\n");
	    break;
        case CUBLAS_STATUS_ARCH_MISMATCH:	
	    printf("The device does not support double-precision.\n");
	    break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
	    printf("The function failed to launch on the GPU.\n");
    }

    cublasDestroy(handle);
}

void spmv( const float *d_inputVector, const int edge, const int m, const float *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, float *d_spmvResult ) {
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

    //cusparseStatus_t status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, alpha, descr, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_inputVector, beta, d_spmvResult);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsrmv(handle,                   
                              CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              m, m, edge, 
                              alpha, descr, 
                              d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                              d_inputVector, beta, d_spmvResult );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //printf("spmv multiplication successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid parameters m, n, or nnz.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("Error: Resources could not be allocated.\n");
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("Error: Device architecture does not support.\n");
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("Error: An internal operation failed.\n");
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("Error: Matrix type not supported.\n");
    }

    // Important: destroy handle
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

__global__ void addResult( int *d_bfsResult, const float *d_spmvResult, const int iter ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //while( tid<N ) {
        d_bfsResult[tid] = (d_spmvResult[tid]>0.5 && d_bfsResult[tid]<0) ? iter : d_bfsResult[tid];
    //    tid += blockDim.x*gridDim.x;
    //}
}

void bfs( const int vertex, const int edge, const int m, const int *d_csrRowPtrA, const int *d_csrColIndA, int *d_bfsResult, const int depth ) {

    // Allocate GPU memory for result
    float *d_spmvResult, *d_spmvSwap;
    cudaMalloc(&d_spmvResult, m*sizeof(float));
    cudaMalloc(&d_spmvSwap, m*sizeof(float));

    // Generate initial vector using vertex
    int *h_bfsResult;
    float *h_spmvResult;
    h_bfsResult = (int*)malloc(m*sizeof(int));
    h_spmvResult = (float*)malloc(m*sizeof(float));

    for( int i=0; i<m; i++ ) {
        h_bfsResult[i]=-1;
        h_spmvResult[i]=0;
        if( i==vertex ) {
            h_bfsResult[i]=0;
            h_spmvResult[i]=1;
        }
    }
    //std::cout << "This is m: " << m << std::endl;
    //print_array(h_bfsResult,m);
    cudaMemcpy(d_spmvSwap,h_spmvResult, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfsResult,h_bfsResult, m*sizeof(int), cudaMemcpyHostToDevice);

    // Generate values for BFS (csrValA where everything is 1)
    float *h_bfsValA, *d_bfsValA;
    h_bfsValA = (float*)malloc(edge*sizeof(float));
    cudaMalloc(&d_bfsValA, edge*sizeof(float));

    for( int i=0; i<edge; i++ ) {
        h_bfsValA[i] = 1;
    }
    cudaMemcpy(d_bfsValA, h_bfsValA, edge*sizeof(float), cudaMemcpyHostToDevice);
    
    // Use Thrust to generate initial vector
    //thrust::device_ptr<float> dev_ptr(d_inputVector);
    //thrust::fill(dev_ptr, dev_ptr + m, (float) 0);
    //thrust::device_vector<float> inputVector(m, 0);
    //inputVector[vertex] = 1;
    //d_inputVector = thrust::raw_pointer_cast( &inputVector[0] );

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    spmv(d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult);
    
    int NBLOCKS = (m+NTHREADS-1)/NTHREADS;

    //axpy(d_spmvSwap, d_bfsValA, m);
    addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, 1 );

    for( int i=2; i<depth; i++ ) {
    //for( int i=2; i<3; i++ ) {
        if( i%2==0 ) {
            spmv( d_spmvResult, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvSwap );
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvSwap, i );
        } else {
            spmv( d_spmvSwap, edge, m, d_bfsValA, d_csrRowPtrA, d_csrColIndA, d_spmvResult );
            addResult<<<NBLOCKS,NTHREADS>>>( d_bfsResult, d_spmvResult, i );
        }
    }

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("GPU BFS finished in %f msec. \n", elapsed);

    //cudaMemcpy(h_spmvResult,d_spmvSwap, m*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_bfsResult,d_bfsResult, m*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_spmvResult,m);
    //print_array(h_bfsResult,m);

    cudaFree(d_spmvResult);
    cudaFree(d_spmvSwap);
    free(h_bfsResult);
    free(h_spmvResult);
}

int main(int argc, char**argv) {
    int m, n, edge;

    // Broken on graphs with more than 500k edges
    freopen(argv[1],"r",stdin);
    //freopen("log","w",stdout);
    printf("Testing %s\n", argv[1]);

    // File i/o
    /*FILE *input = fopen(argv[1], "r");

    bool directed;
    int c = fgetc(input);
    int old_c = 0;

    //printf("%d\n",c);
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, input);
            //printf("%d %d\n",old_c,c);
            break;
        }
        old_c = c;
        c = fgetc(input);
    }
    fscanf(input, "%d %d %d", &m, &n, &edge);
    
    // Allocate memory depending on how many edges are present
    float *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    h_csrValA    = (float*)malloc(edge*sizeof(float));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( fscanf(input, "%d", &h_csrColIndA[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        fscanf(input, "%d", &h_cooRowIndA[j]);

        if( j==0 ) {
            c=fgetc(input);
            //printf("c = %d\n",c);
        }

        if( c!=32 ) {
            h_csrValA[j]=1.0;
            if( j==0 ) directed = false;
        } else {
            fscanf(input, "%f", &h_csrValA[j]);
        }

        h_cooRowIndA[j]--;
        h_csrColIndA[j]--;
        //printf("%d %d %d\n", h_cooRowIndA[j], h_csrColIndA[j], j);
    }
    fclose(input);
    if( directed==true ) {
        printf("The graph is directed: ");
        print_end(h_csrValA,edge);
    } else {
        printf("The graph is undirected.\n");
    }*/

    bool directed;
    int c = getchar();
    int old_c = 0;
    //printf("%d\n",c);
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, stdin);
            //printf("%d %d\n",old_c,c);
            break;
        }
        old_c = c;
        c=getchar();
    }
    scanf("%d %d %d", &m, &n, &edge);
    
    // Allocate memory depending on how many edges are present
    float *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    h_csrValA    = (float*)malloc(edge*sizeof(float));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_csrColIndA[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowIndA[j]);

        if( j==0 ) {
            c=getchar();
            //printf("c = %d\n",c);
        }

        if( c!=32 ) {
            h_csrValA[j]=1.0;
            if( j==0 ) directed = false;
        } else {
            scanf("%f", &h_csrValA[j]);
        }

        h_cooRowIndA[j]--;
        h_csrColIndA[j]--;
        //printf("%d %d %d\n", h_cooRowIndA[j], h_csrColIndA[j], j);
    }
    if( directed==true ) {
        printf("The graph is directed: ");
        print_end(h_csrValA,edge);
    } else {
        printf("The graph is undirected.\n");
    }

    // Allocate GPU memory
    float *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    float *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;
    int *d_bfsResult;
    cudaMalloc(&d_bfsResult, m*sizeof(int));

    cudaMalloc(&d_csrValA, edge*sizeof(float));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    cudaMalloc(&d_cscValA, edge*sizeof(float));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(h_cooRowIndA, d_cooRowIndA, (edge)*sizeof(int),cudaMemcpyDeviceToHost);
    //print_vector(h_cooRowIndA,edge);

    // Run COO -> CSR kernel
    coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // Run BFS on CPU. Need data in CSR form first.
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    //print_array(h_csrRowPtrA,m+1);

    int depth = 1000;
    depth = bfsCPU( 0, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, depth );

    // Some testing code. To be turned into unit test.
    //int depth = 4;
    //bfsCPU( 0, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, depth );
    //depth++;
    print_end_interesting(h_bfsResultCPU, m);

    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();
    // Run CSR -> CSC kernel
    csr2csc( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    gpu_timer2.Start();

    // Run BFS kernel on GPU
    // Non-transpose spmv
    //bfs( i, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, 5 );
    //bfs( 0, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_bfsResult, 5 );

    bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth );
    gpu_timer.Stop();
    gpu_timer2.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    printf("GPU BFS finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    printf("GPU BFS finished in %f msec. not including transpose\n", elapsed2);

    // Run check for errors
    cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    verify( m, h_bfsResult, h_bfsResultCPU );
    print_array(h_bfsResult, m);

    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_cooRowIndA);

    cudaFree(d_cscValA);
    cudaFree(d_cscRowIndA);
    cudaFree(d_cscColPtrA);
    cudaFree(d_bfsResult);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooRowIndA);
    free(h_bfsResult);
    free(h_bfsResultCPU);

    //free(h_cscValA);
    //free(h_cscRowIndA);
    //free(h_cscColPtrA);
}
