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
#include <moderngpu.cuh>

#include <util.cuh>
#include <tc.cuh>

#include <testBfs.cpp>
#include <string.h>

void runBfs(int argc, char**argv) { 
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
    bool undirected = false;
    if( parseArgs( argc, argv, source, device, delta, undirected )==true ) {
        printf( "Usage: test apple.mtx -source 5\n");
        return;
    }
    //cudaSetDevice(device);
    printf("Testing %s from source %d\n", argv[1], source);
    
    // 2. Reads in number of edges, number of nodes
    readEdge( m, n, edge, stdin );
    printf("Graph has %d nodes, %d edges\n", m, edge);

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_cscValA;
    //int *h_csrRowPtrA, *h_csrColIndA;
    int *h_cscRowIndA, *h_cscColPtrA;
    int *h_cooColIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    //h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cscValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    //h_cooValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    //h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    //h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cscColPtrA = (int*)malloc((m+1)*sizeof(int));
    h_cscRowIndA = (int*)malloc(edge*sizeof(int));
    //h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_cooColIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // 3b. Allocate memory to second and third matrices
    int edge_B, edge_C;
    int m_B, m_C;
    edge_B = edge;
    edge_C = 2*edge;
    m_B = m;
    m_C = m;

    typeVal *h_cscValB, *h_cscValC;
    int *h_cscRowIndB, *h_cscColPtrB;
    int *h_cscRowIndC, *h_cscColPtrC;
    h_cscValB = (typeVal*)malloc(edge_B*sizeof(typeVal));
    //h_cscValC = (typeVal*)malloc(edge_C*sizeof(typeVal));
    h_cscRowIndB = (int*)malloc(edge_B*sizeof(int));
    //h_cscRowIndC = (int*)malloc(edge_C*sizeof(int));
    h_cscColPtrB = (int*)malloc(m_B*sizeof(int));
    h_cscColPtrC = (int*)malloc(m_C*sizeof(int));

    // 4. Read in graph from .mtx file
    // We are actually doing readMtx<typeVal>( edge, h_cooColIndA, h_cscRowIndA, h_cscValA );
    readMtx<typeVal>( edge, h_cscRowIndA, h_cooColIndA, h_cscValA );
    print_array( h_cooColIndA, m );

    // 5. Allocate GPU memory
    typeVal *d_cscValA, *d_cscValB, *d_cscValC;
    typeVal *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA;
    int *d_cscRowIndA, *d_cscColPtrA;
    int *d_cscRowIndB, *d_cscColPtrB;
    int *d_cscRowIndC, *d_cscColPtrC;
    int *d_cooColIndA;
    int *d_bfsResult;
    cudaMalloc(&d_bfsResult, m*sizeof(int));

    cudaMalloc(&d_csrValA, edge*sizeof(typeVal));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    //cudaMalloc(&d_cooRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cooColIndA, edge*sizeof(int));
    cudaMalloc(&d_cscValA, edge*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // 5b GPU memory for matrices B and C
    cudaMalloc(&d_cscValB, edge_B*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndB, edge_B*sizeof(int));
    cudaMalloc(&d_cscColPtrB, (m_B+1)*sizeof(int));
    //cudaMalloc(&d_cscValC, edge_C*sizeof(typeVal)); // Allocate C in mXm
    //cudaMalloc(&d_cscRowIndC, edge_C*sizeof(int));
    //cudaMalloc(&d_cscColPtrC, (m_C+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_cscValA, h_cscValA, (edge)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIndA, h_cscRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooColIndA, h_cooColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. Run COO -> CSR kernel
    // We are actually doing coo2csc( d_cooColIndA, edge, m, d_cscColPtrA )
    coo2csr( d_cooColIndA, edge, m, d_cscColPtrA );
    cudaMemcpy(d_cscValB, d_cscValA, edge*sizeof(typeVal), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cscRowIndB, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cscColPtrB, d_cscColPtrA, (m+1)*sizeof(int), cudaMemcpyDeviceToDevice);

    // 8. Run BFS on CPU. Need data in CSR form first.
    cudaMemcpy(h_cscColPtrA,d_cscColPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    int depth = 1000;
    depth = bfsCPU( source, m, h_cscColPtrA, h_cscRowIndA, h_bfsResultCPU, depth );
    print_end_interesting(h_bfsResultCPU, m);

    // Make two GPU timers
    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();

    // 9. Run CSR -> CSC kernel
    csr2csc<typeVal>( m, edge, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_csrValA, d_csrColIndA, d_csrRowPtrA );
    gpu_timer.Stop();
    gpu_timer2.Start();

    // 10. Run BFS kernel on GPU
    for( int i=0; i<m+1; i++ )
        h_cscColPtrB[i] = h_cscColPtrA[i];
    cudaMemcpy(h_cscRowIndA, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtrA, d_cscColPtrA, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValA, d_cscValA, edge*sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix( h_cscValA, h_cscColPtrA, h_cscRowIndA, m );
    cudaMemcpy(h_cscRowIndB, d_cscRowIndB, edge*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtrB, d_cscColPtrB, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValB, d_cscValB, edge*sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix( h_cscValB, h_cscColPtrB, h_cscRowIndB, m );
    //mXm<typeVal>( edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_cscValB, h_cscColPtrB, d_cscColPtrB, d_cscRowIndB, d_cscValC, d_cscColPtrC, d_cscRowIndC, *context);
    //bfs<typeVal>( source, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, *context );
    edge_C = spgemm<typeVal>( edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_cscValB, d_cscColPtrB, d_cscRowIndB, d_cscValC, d_cscColPtrC, d_cscRowIndC );

    gpu_timer2.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    elapsed2 += gpu_timer2.ElapsedMillis();

    printf("CSR->CSC finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    printf("GPU BFS finished in %f msec. not including transpose\n", elapsed2);

    cudaMemcpy(h_cscRowIndA, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_cscRowIndA, m);

    // Compare with CPU BFS for errors
    //cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    //verify( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);
    h_cscValC = (typeVal*)malloc(edge_C*sizeof(typeVal));
    h_cscRowIndC = (int*)malloc(edge_C*sizeof(int));
    cudaMemcpy(h_cscRowIndC, d_cscRowIndC, edge_C*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtrC, d_cscColPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValC, d_cscValC, edge_C*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Matrix C: %dx%d with %d nnz\n", m, m, edge_C);
    print_matrix( h_cscValC, h_cscColPtrC, h_cscRowIndC, m );

    // Compare with SpMV for errors
    //bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_bfsResult, depth, *context);
    //cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    //verify( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);
}

int main(int argc, char**argv) {
    runBfs(argc, argv);
}    
