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
#include <queue>
#include <cusparse.h>
#include <moderngpu.cuh>

#include <util.cuh>
#include <sssp.cuh>

#include <string.h>
#include <testBfs.cpp>

#define MARK_PREDECESSORS 0

class CompareDist {
public:
    bool operator() ( const std::pair<int, float>& lhs, const std::pair<int, float>& rhs ) const {
        return lhs.second > rhs.second;
    }
};

template<typename T> void print_queue(T& q, int m) {
    int count = 0;
    std::pair<int, float> Edge;
    while(!q.empty() && count<m ) {
        //Edge = q.top();
        printf("[%d]: %f ", q.top().first, q.top().second);
        q.pop();
        count++;
    }
    printf("\n");
}

// A simple CPU-based reference SSSP ranking implementation
template<typename VertexId, typename value>
int SimpleReferenceSssp(
    const VertexId m, const VertexId *h_rowPtrA, const VertexId *h_colIndA, const value *h_csrValA,
    value                                   *source_path,
    VertexId                                *predecessor,
    VertexId                                src,
    VertexId                                stop)
{
    typedef std::pair<VertexId, value> Edge;

    // Initialize queue for managing previously-discovered nodes
    std::priority_queue<std::pair<VertexId, value>, std::vector<std::pair<VertexId, value> >, CompareDist> frontier;

    //initialize distances
    //  use -1 to represent infinity for source_path
    //                      undefined for predecessor
    for (VertexId i = 0; i < m; ++i) {
        source_path[i] = -1;
        //Edge = std::make_pair(i, h_csrValA[i]);
        frontier.push(std::pair<VertexId, value>(i, h_csrValA[i]));
        if (MARK_PREDECESSORS)
            predecessor[i] = -1;
    }
    source_path[src] = 0;
    frontier.push(std::pair<VertexId, value>(src, 0));
    VertexId search_depth = 0;

    //print_queue(frontier, 10);

    //
    //Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty()) {
        
        // Dequeue node from frontier
        Edge dequeued_node = frontier.top();
        frontier.pop();

        // Set v as vertex index, d as distance
        VertexId v = dequeued_node.first;
        value d = dequeued_node.second;

        // Locate adjacency list
        int edges_begin = h_rowPtrA[v];
        int edges_end = h_rowPtrA[v+1];

        // Checks that we only iterate through once
        //   -necessary because we will be having redundant vertices in
        //   queue so we will only do work when we have the best one
        //   -source_path[v] == -1 means we haven't explored it before
        if( source_path[v] == -1 || d <= source_path[v] ) {
            for( int edge = edges_begin; edge < edges_end; ++edge ) {
                //Lookup neighbor and enqueue if undiscovered
                VertexId neighbor = h_colIndA[edge];
                value alt_dist = source_path[v] + h_csrValA[edge];
                if( source_path[neighbor] == -1 || alt_dist < source_path[neighbor] ) {
                    source_path[neighbor] = alt_dist;
                    frontier.push(std::pair<VertexId,value>(neighbor,alt_dist));
                    if(MARK_PREDECESSORS) 
                        predecessor[neighbor] = dequeued_node.first;
                }
            }
        }
    }
    
    if (MARK_PREDECESSORS)
        predecessor[src] = -1;

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    printf("CPU SSSP finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

    return search_depth;
}

int ssspCPU( const int src, const int m, const int *h_rowPtrA, const int *h_colIndA, const float* h_csrValA, float *h_ssspResultCPU, const int stop ) {

    typedef int VertexId; // Use as the node identifier type
    typedef float value;

    VertexId *reference_check_preds = NULL;

    int depth = SimpleReferenceSssp<VertexId, value>(
        m, h_rowPtrA, h_colIndA, h_csrValA,
        h_ssspResultCPU,
        reference_check_preds,
        src,
        stop);

    print_array(h_ssspResultCPU, m);
    return depth;
}

void runSssp(int argc, char**argv) { 
    int m, n, edge;
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);

    // Define what filetype edge value should be stored
    typedef float typeVal;

    // File i/o
    // 1. Open file from command-line 
    // -source 1
    freopen(argv[1],"r",stdin);
    int source;
    int device;
    float delta;
    if( parseArgs( argc, argv, source, device, delta )==true ) {
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
    float *h_ssspResult, *h_ssspResultCPU;

    h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_ssspResult = (float*)malloc((m)*sizeof(float));
    h_ssspResultCPU = (float*)malloc((m)*sizeof(float));

    // 4. Read in graph from .mtx file
    readMtx<typeVal>( edge, h_csrColIndA, h_cooRowIndA, h_csrValA );
    print_array( h_cooRowIndA, m );

    // 5. Allocate GPU memory
    typeVal *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    typeVal *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;
    float *d_ssspResult;
    cudaMalloc(&d_ssspResult, m*sizeof(float));

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

    // 8. Run SSSP on CPU. Need data in CSR form first.
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    int depth = 1000;
    depth = ssspCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_ssspResultCPU, depth );
    print_end_interesting(h_ssspResultCPU, m);

    // Verify SSSP CPU with BFS CPU.
    bfsCPU<float>( source, m, h_csrRowPtrA, h_csrColIndA, h_ssspResult, 1000);
    verify<float>( m, h_ssspResultCPU, h_ssspResult );

    // Make two GPU timers
    /*GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();

    // 9. Run CSR -> CSC kernel
    csr2csc<typeVal>( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    gpu_timer.Stop();
    gpu_timer2.Start();

    // 10. Run SSSP kernel on GPU
    //sssp<typeVal>( source, edge, m, d_csrValA, d_cscColPtrA, d_cscRowIndA, d_ssspResult, depth, *context );
    sssp<typeVal>( source, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_ssspResult, depth, *context );

    gpu_timer2.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    printf("CSR->CSC finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    //printf("GPU SSSP finished in %f msec. not including transpose\n", elapsed2);

    cudaMemcpy(h_csrColIndA, d_csrColIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_csrColIndA, m);

    // Compare with CPU SSSP for errors
    cudaMemcpy(h_ssspResult,d_ssspResult,m*sizeof(float),cudaMemcpyDeviceToHost);
    verify<float>( m, h_ssspResult, h_ssspResultCPU );
    print_array(h_ssspResult, m);

    // Compare with SpMV for errors
    //bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth, *context);
    //cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    //verify<int>( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);
    
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_cooRowIndA);

    cudaFree(d_cscValA);
    cudaFree(d_cscRowIndA);
    cudaFree(d_cscColPtrA);
    cudaFree(d_ssspResult);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooRowIndA);
    free(h_ssspResult);
    free(h_ssspResultCPU);

    //free(h_cscValA);
    //free(h_cscRowIndA);
    //free(h_cscColPtrA);*/
}

int main(int argc, char**argv) {
    runSssp(argc, argv);
}    
