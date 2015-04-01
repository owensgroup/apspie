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
#include <curand.h>
#include <cusparse.h>
#include <moderngpu.cuh>

#include <util.cuh>
#include <spmspvMis.cuh>

#include <string.h>


// A simple CPU-based reference MIS ranking implementation
template<typename VertexId>
int SimpleReferenceMis(
    const VertexId m, const VertexId *h_rowPtrA, const VertexId *h_colIndA,
    VertexId                                *source_path,
    VertexId                                src)
{
    //initialize distances
    for (VertexId i = 0; i < m; ++i) {
        source_path[i] = -1;
    }
    source_path[src] = 1;
    int edges_begin = h_rowPtrA[src];
    int edges_end = h_rowPtrA[src + 1];

    for( int edge=edges_begin; edge<edges_end; edge++ ) {
        VertexId neighbor = h_colIndA[edge];

        if (source_path[neighbor] == -1)
            source_path[neighbor] = 0;
    }
    
    VertexId search_depth = 1;

    //
    //Perform MIS
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
   
    for( VertexId i=0; i<m; i++ ) {
        if( source_path[i]==-1 ) {
            source_path[i] = 1;
            
            // Locate adjacency list 
            edges_begin = h_rowPtrA[i];
            edges_end = h_rowPtrA[i + 1];

            /*for( int edge=edges_begin; edge<edges_end; edge++ ) {
                VertexId neighbor = h_colIndA[edge];
                if( neighbor==i ) {
                    flag = 1;
                    source_path[i] = 0;
                    break;
                }
            }
            if( flag!=1 )*/
                for( int edge=edges_begin; edge<edges_end; edge++ ) {
                    VertexId neighbor = h_colIndA[edge];

                    if( source_path[neighbor]==-1 )
                        source_path[neighbor] = 0;
             }
        }
    }
 
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    printf("CPU MIS finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

    return search_depth;
}

int misCPU( const int src, const int m, const int *h_rowPtr, const int *h_colInd, int *h_misResultCPU ) {

    typedef int VertexId; // Use as the node identifier type

    int depth = SimpleReferenceMis<VertexId>(
        m, h_rowPtr, h_colInd,
        h_misResultCPU,
        src);

    //print_array(h_misResultCPU, m);
    return depth;
}

void verifyMis( const int edge, const int m, const int *d_misResultCPU, const int *d_csrRowPtr, const int *d_csrColInd, int *d_result, mgpu::CudaContext& context ) {
    spmspvCsr<int>( d_misResultCPU, edge, m, d_csrRowPtr, d_csrColInd, d_result, context );
}

void fillUniform( float *d_A, int edge ) {
    curandGenerator_t prng;
    curandCreateGenerator( &prng, CURAND_RNG_PSEUDO_DEFAULT );

    curandGenerateUniform( prng, d_A, edge );
}

template<typename typeVal>
int makeSymmetric( int edge, int *h_csrColIndA, int *h_cooRowIndA, typeVal *h_randVec ) {

    int realEdge = edge/2;
    
    for( int i=0; i<realEdge; i++ ) {
        h_cooRowIndA[realEdge+i] = h_csrColIndA[i];
        h_csrColIndA[realEdge+i] = h_cooRowIndA[i];
    }

    // Sort
    //struct arrayset *work = (arrayset*)malloc(edge*sizeof(arrayset));
    //work->values1 = h_cooRowIndA;
    //work->values2 = h_csrColIndA;
    struct arrayset work = { h_cooRowIndA, h_csrColIndA };
    custom_sort(&work, edge);

    int curr = h_csrColIndA[0];
    int last;
    int curr_row = h_cooRowIndA[0];
    int last_row;
    if( curr_row == curr )
        h_cooRowIndA[0] = -1;

    // Check for self-loops and repetitions, mark with -1
    for( int i=1; i<edge; i++ ) {
        last = curr;
        last_row = curr_row;
        curr = h_csrColIndA[i];
        curr_row = h_cooRowIndA[i];

        // Self-loops
        if( curr_row == curr )
            h_csrColIndA[i] = -1;
        // Repetitions
        else if( curr == last && curr_row == last_row )
            h_csrColIndA[i] = -1;
    }

    // Remove self-loops and repetitions.
    int shift = 0;
    int back = 0;
    for( int i=0; i+shift<edge; i++ ) {
        if(h_csrColIndA[i] == -1) {
            for( shift; back<=edge; shift++ ) {
                back = i+shift;
                if( h_csrColIndA[back] != -1 ) {
                    //printf("Swapping %d with %d\n", i, back ); 
                    h_csrColIndA[i] = h_csrColIndA[back];
                    h_cooRowIndA[i] = h_cooRowIndA[back];
                    h_csrColIndA[back] = -1;
                    break;
    }}}}
    return edge-shift;
}

void runMis(int argc, char**argv) { 
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

    // Double # of edges because symmetric/undirected
    edge *= 2;

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_randVec;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;
    int *h_misResult, *h_misResultCPU;

    h_randVec    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_misResult = (int*)malloc((m)*sizeof(int));
    h_misResultCPU = (int*)malloc((m)*sizeof(int));

    // 4. Read in graph from .mtx file
    readMtx<typeVal>( edge/2, h_csrColIndA, h_cooRowIndA, h_randVec );
    //print_array( h_cooRowIndA, 40 );
    edge = makeSymmetric( edge, h_csrColIndA, h_cooRowIndA, h_randVec );
    printf("Undirected graph has %d edges\n", edge);

    // 5. Allocate GPU memory
    typeVal *d_randVec;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    int *d_misResult;
    cudaMalloc(&d_misResult, m*sizeof(int));

    cudaMalloc(&d_randVec, edge*sizeof(typeVal));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_randVec, h_randVec, (edge)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. Run COO -> CSR kernel
    coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // 8. Run MIS on CPU. Need data in CSR form first.
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    misCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_misResultCPU );
    print_end_interesting(h_misResultCPU, m);

    // 9. Verify CPU-MIS by running BFS 1x on GPU.
    cudaMemcpy( d_misResult, h_misResultCPU, m*sizeof(int), cudaMemcpyHostToDevice );
    verifyMis( edge, m, d_misResult, d_csrRowPtrA, d_csrColIndA, d_cooRowIndA, *context );
    cudaMemcpy( h_misResult, d_cooRowIndA, m*sizeof(int), cudaMemcpyDeviceToHost );
    unverify( m, h_misResult, h_misResultCPU );

    // Make two GPU timers
    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();

    // 9. Generate random numbers
    //mis( i, edge, m, d_randVec, d_csrRowPtrA, d_csrColIndA, d_misResult, 5 );
    fillUniform( d_randVec, m);
    cudaMemcpy( h_randVec, d_randVec, m*sizeof(typeVal), cudaMemcpyDeviceToHost );
    //print_array( h_randVec, 40);

    // 10. Run MIS kernel on GPU
    spmspvMis( edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_randVec, d_misResult, delta, *context); 
    //mis( edge, m, d_csrRowPtrA, d_csrColIndA, d_misResult, delta, *context);
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    printf("using step-size %f in %f ms\n", delta, elapsed);
    //printf("GPU MIS finished in %f msec. not including transpose\n", elapsed2);

    //cudaMemcpy(h_csrColIndA, d_csrColIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    //print_array(h_csrColIndA, m);

    // 11. Verify GPU-MIS by running BFS 1x on GPU
    cudaMemcpy( h_misResultCPU, d_misResult, m*sizeof(int), cudaMemcpyDeviceToHost );
    verifyMis( edge, m, d_misResult, d_csrRowPtrA, d_csrColIndA, d_cooRowIndA, *context );
    cudaMemcpy( h_misResult, d_cooRowIndA, m*sizeof(int), cudaMemcpyDeviceToHost );
    unverify( m, h_misResult, h_misResultCPU );

    // Compare with SpMV for errors
    //cuspMis( 0, edge, m, d_csrRowPtrA, d_csrColIndA, d_misResult, depth, *context);
    //cudaMemcpy(h_misResult,d_misResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    //verify( m, h_misResult, h_misResultCPU );
    //print_array(h_misResult, m);
    
    cudaFree(d_randVec);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_cooRowIndA);
    cudaFree(d_misResult);

    free(h_randVec);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooRowIndA);
    free(h_misResult);
    free(h_misResultCPU);

}

int main(int argc, char**argv) {
    runMis(argc, argv);
}    
