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

#include <string.h>

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

void coo2csr( const int *d_cooRowIndA, const int edge, const int m, int *d_csrRowPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("COO->CSR finished in %f msec. \n", elapsed);

    // Important: destroy handle
    cusparseDestroy(handle);
}

template<typename typeVal>
void csr2csc( const int m, const int edge, const typeVal *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, typeVal *d_cscValA, int *d_cscRowIndA, int *d_cscColPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // For CUDA 4.0
    //cusparseStatus_t status = cusparseScsr2csc(handle, m, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, 1, CUSPARSE_INDEX_BASE_ZERO);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsr2csc(handle, m, m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);

    // Important: destroy handle
    cusparseDestroy(handle);
}

// This function extracts the number of nodes and edges from input file
void readEdge( int &m, int &n, int &edge, FILE *inputFile ) {
    int c = getchar();
    int old_c = 0;
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, inputFile);
            break;
        }
        old_c = c;
        c=getchar();
    }
    scanf("%d %d %d", &m, &n, &edge);
}

// This function loads a graph from .mtx input file
template<typename typeVal>
void readMtx( int edge, int *h_csrColInd, int *h_cooRowInd, typeVal *h_csrVal ) {
    bool weighted = true;
    int c;
    int csr_max = 0;
    int csr_current = 0;
    int csr_row = 0;
    int csr_first = 1;

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_csrColInd[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowInd[j]);

        if( j==0 ) {
            c=getchar();
        }

        if( c!=32 ) {
            h_csrVal[j]=1.0;
            if( j==0 ) weighted = false;
        } else {
            //std::cin >> h_csrVal[j];
            scanf("%f", &h_csrVal[j]);
        }

        h_cooRowInd[j]--;
        h_csrColInd[j]--;

        // Finds max csr row.
        if( j!=0 ) {
            if( h_cooRowInd[j]==0 ) csr_first++;
            if( h_cooRowInd[j]==h_cooRowInd[j-1] )
                csr_current++;
            else {
                if( csr_current > csr_max ) {
                    csr_max = csr_current;
                    csr_current = 1;
                    csr_row = h_cooRowInd[j-1];
                }
            }
        }
    }
    printf("The biggest row was %d with %d elements.\n", csr_row, csr_max);
    printf("The first row has %d elements.\n", csr_first);
    if( weighted==true ) {
        printf("The graph is weighted. ");
    } else {
        printf("The graph is unweighted.\n");
    }
}

bool parseArgs( int argc, char**argv, int &source, int &device ) {
    bool error = false;
    source = 0;
    device = 0;

    if( argc%2!=0 )
        return true;   
 
    for( int i=2; i<argc; i+=2 ) {
       if( strstr(argv[i], "-source") != NULL )
           source = atoi(argv[i+1]);
       else if( strstr(argv[i], "-device") != NULL )
           device = atoi(argv[i+1]);
    }
    return error;
}

void runBfs(int argc, char**argv) { 
    int m, n, edge;
    ContextPtr context = mgpu::CreateCudaDevice(0);

    // Define what filetype edge value should be stored
    typedef float typeVal;

    // File i/o
    // 1. Open file from command-line 
    // -source 1
    freopen(argv[1],"r",stdin);
    int source;
    int device;
    if( parseArgs( argc, argv, source, device )==true ) {
        printf( "Usage: test apple.mtx -source 5\n");
        return;
    }
    //cudaSetDevice(device);
    printf("Testing %s from source %d\n", argv[1], source);
    
    // 2. Reads in number of edges, number of nodes
    readEdge( m, n, edge, stdin );
    printf("Graph has %d nodes, %d edges\n", m, edge);

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // 4. Read in graph from .mtx file
    readMtx<typeVal>( edge, h_csrColIndA, h_cooRowIndA, h_csrValA );
    print_array( h_csrRowPtrA, m );

    // 5. Allocate GPU memory
    typeVal *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    typeVal *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;
    int *d_bfsResult;
    cudaMalloc(&d_bfsResult, m*sizeof(int));

    cudaMalloc(&d_csrValA, edge*sizeof(typeVal));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    cudaMalloc(&d_cscValA, edge*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. Run COO -> CSR kernel
    coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // 8. Run BFS on CPU. Need data in CSR form first.
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    int depth = 1000;
    depth = bfsCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, depth );
    print_end_interesting(h_bfsResultCPU, m);

    // Make two GPU timers
    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();

    // 9. Run CSR -> CSC kernel
    csr2csc<typeVal>( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    gpu_timer.Stop();
    gpu_timer2.Start();

    // 10. Run BFS kernel on GPU
    //bfs( i, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, 5 );
    //bfs( 0, edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_bfsResult, 5 );

    // 10. Run BFS kernel on GPU
    spsvBfs( source, edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, *context); 
    //bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth, *context);
    gpu_timer2.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    printf("CSR->CSC finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    //printf("GPU BFS finished in %f msec. not including transpose\n", elapsed2);

    cudaMemcpy(h_csrColIndA, d_csrColIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrColIndA, m);

    // Compare with CPU BFS for errors
    cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    verify( m, h_bfsResult, h_bfsResultCPU );
    print_array(h_bfsResult, m);

    // Compare with SpMV for errors
    bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth, *context);
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

int main(int argc, char**argv) {
    runBfs(argc, argv);
}    
