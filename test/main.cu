// Puts everything together
// For now, just run V times.
// Optimizations: 
// -come up with good stopping criteria [done]
// -start from i=1 [done]
// -test whether float really are faster than ints
// -distributed idea
// -change nthread [done - doesn't work]
 
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <deque>
#include <cusparse.h>

#include <util.cuh>
#include <bfs.cuh>
#include <spsvBfs.cuh>

#define MARK_PREDECESSORS 0

// A simple CPU-based reference BFS ranking implementation
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

    //printf("CPU BFS finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

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

void coo2csr( const int *d_cooRowIndA, const int edge, const int m, int *d_csrRowPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    //printf("COO->CSR finished in %f msec. \n", elapsed);

    /*switch( status ) {
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
    }*/

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

    /*switch( status ) {
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
    }*/

    // Important: destroy handle
    cusparseDestroy(handle);
}

int main(int argc, char**argv) {
    int m, n, edge;
    
    ContextPtr context = mgpu::CreateCudaDevice(0);

    // Broken on graphs with more than 500k edges
    freopen(argv[1],"r",stdin);
    //freopen("log","w",stdout);
    //printf("Testing %s\n", argv[1]);

    // File i/o

    bool weighted = true;
    int c = getchar();
    int old_c = 0;
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, stdin);
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

    int csr_max = 0;
    int csr_current = 0;
    int csr_row = 0;
    int csr_first = 1;

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_csrColIndA[j])==EOF ) {
            //printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowIndA[j]);

        if( j==0 ) {
            c=getchar();
        }

        if( c!=32 ) {
            h_csrValA[j]=1.0;
            if( j==0 ) weighted = false;
        } else {
            scanf("%f", &h_csrValA[j]);
        }

        h_cooRowIndA[j]--;
        h_csrColIndA[j]--;

        // Finds max csr row.
        if( j!=0 ) {
            if( h_cooRowIndA[j]==0 ) csr_first++;
            if( h_cooRowIndA[j]==h_cooRowIndA[j-1] )
                csr_current++;
            else {
                if( csr_current > csr_max ) {
                    csr_max = csr_current;
                    csr_current = 1;
                    csr_row = h_cooRowIndA[j-1];
                }
            }
        }
    }
    //printf("The biggest row was %d with %d elements.\n", csr_row, csr_max);
    //printf("The first row has %d elements.\n", csr_first);
    if( weighted==true ) {
        //printf("The graph is weighted. ");
        //print_end(h_csrValA,edge);
    } else {
        //printf("The graph is unweighted.\n");
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

    int depth[10];
    for(int i=0;i<10;i++) depth[i] = bfsCPU( i, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, 1000 );

    // Some testing code. To be turned into unit test.
    //int depth = 4;
    //bfsCPU( 0, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, depth );
    //depth++;
    //print_end_interesting(h_bfsResultCPU, m);

    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();
    // Run CSR -> CSC kernel
    csr2csc( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    gpu_timer.Stop();
    gpu_timer2.Start();

    // Run BFS kernel on GPU
    // Non-transpose spmv
    //bfs( i, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, 5 );
    //bfs( 0, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_bfsResult, 5 );

    bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth, *context);
    //cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    //verify( m, h_bfsResult, h_bfsResultCPU );
    
    // 0-no sort (pigeonhole/address sort)
    // 1-merge sort
    // 2-radix sort
    spsvBfs( 0, edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, 0, *context); 
    spsvBfs( 0, edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, 1, *context); 
    spsvBfs( 0, edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, 2, *context); 

    gpu_timer2.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    //printf("CSR->CSC finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    //printf("GPU BFS finished in %f msec. not including transpose\n", elapsed2);

    //cudaMemcpy(h_csrColIndA, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIndA, d_csrColIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_csrColIndA, m);

    // Run check for errors
    cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    verify( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);

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
