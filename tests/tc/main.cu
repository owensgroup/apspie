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


int countDiag( const int edge, const int *h_cscRowIndA, const int* h_cooColIndA ) {
    int count = 0;
    for( int i=0; i<edge; i++ )
        if( h_cscRowIndA[i]==h_cooColIndA[i] )
            count++;
    return count;
}

template< typename T >
void buildVal( const int edge, T *h_cscValA ) {
    for( int i=0; i<edge; i++ )
        h_cscValA[i] = 1.0;
}

template< typename T >
void buildLower( const int m, const int edge_B, const int *h_cscColPtrA, const int *h_cscRowIndA, const T *h_cscValA, int *h_cscColPtrB, int *h_cscRowIndB, T *h_cscValB, bool lower=true ) {

    int count = 0;
    h_cscColPtrB[0] = count;
    for( int i=0; i<m; i++ ) {
        for( int j=h_cscColPtrA[i]; j<h_cscColPtrA[i+1]; j++ ) {
            if( (lower==true && h_cscRowIndA[j] > i) || (lower==false && h_cscRowIndA[j] < i) ) {
                printf("%d %d %d\n", i, j, count);
                h_cscRowIndB[count] = h_cscRowIndA[j];
                h_cscValB[count] = h_cscValA[j];
                count++;
            }
        }
        h_cscColPtrB[i+1] = count;
    }
}

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
    //if( undirected ) edge=2*edge;

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_cscValA, *h_cooValA;
    //int *h_csrRowPtrA, *h_csrColIndA;
    int *h_cscRowIndA, *h_cscColPtrA;
    int *h_cooRowIndA, *h_cooColIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    //h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cscValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cooValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    //h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    //h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cscColPtrA = (int*)malloc((m+1)*sizeof(int));
    h_cscRowIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_cooColIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // 4. Read in graph from .mtx file
    // We are actually doing readMtx<typeVal>( edge, h_cooColIndA, h_cscRowIndA, h_cscValA );
    CpuTimer cpu_timerRead;
    CpuTimer cpu_timerMake;
    CpuTimer cpu_timerBuild;
    printf("Old edge #: %d\n", edge);
    cpu_timerRead.Start();
    readMtx<typeVal>( edge/2, h_cooColIndA, h_cooRowIndA, h_cooValA );
    cpu_timerRead.Stop();
    cpu_timerMake.Start();
    edge = makeSymmetric( edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
    cpu_timerMake.Stop();
    printf("\nUndirected graph has %d nodes, %d edges\n", m, edge);
    cpu_timerBuild.Start();
    buildMatrix<typeVal>( h_cscColPtrA, h_cscRowIndA, h_cscValA, m, edge, h_cooRowIndA, h_cooColIndA, h_cooValA );
    cpu_timerBuild.Stop();
    float elapsedRead = cpu_timerRead.ElapsedMillis();
    float elapsedMake = cpu_timerMake.ElapsedMillis();
    float elapsedBuild= cpu_timerBuild.ElapsedMillis();
    printf("readMtx: %f ms\n", elapsedRead);
    printf("makeSym: %f ms\n", elapsedMake);
    printf("buildMat: %f ms\n", elapsedBuild);

    // 4b. Count diagonal
    int diag = countDiag( edge, h_cooRowIndA, h_cooColIndA ); 
    int edge_B = (edge-diag)/2;
    int edge_C = edge_B;
    printf("Number of elements on diagonal: %d\n", diag);
    printf("Number of elements on L: %d\n", edge_B);

    // 4c. Allocate memory to second and third matrices
    int m_B = m;
    int m_C = m;

    typeVal *h_cscValB, *h_cscValC;
    int *h_cscRowIndB, *h_cscColPtrB;
    int *h_cscRowIndC, *h_cscColPtrC;
    h_cscValB = (typeVal*)malloc(edge_B*sizeof(typeVal));
    h_cscValC = (typeVal*)malloc(edge_C*sizeof(typeVal));
    h_cscRowIndB = (int*)malloc(edge_B*sizeof(int));
    h_cscRowIndC = (int*)malloc(edge_C*sizeof(int));
    h_cscColPtrB = (int*)malloc(m_B*sizeof(int));
    h_cscColPtrC = (int*)malloc(m_C*sizeof(int));

    buildVal( edge, h_cscValA );
    buildLower( m, edge_B, h_cscColPtrA, h_cscRowIndA, h_cscValA, h_cscColPtrB, h_cscRowIndB, h_cscValB );
    //print_matrix( h_cscValA, h_cscColPtrA, h_cscRowIndA, m );
    //print_matrix( h_cscValB, h_cscColPtrB, h_cscRowIndB, m );
    buildLower( m, edge_C, h_cscColPtrA, h_cscRowIndA, h_cscValA, h_cscColPtrC, h_cscRowIndC, h_cscValC, false );
    //print_matrix( h_cscValC, h_cscColPtrC, h_cscRowIndC, m );

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

    //cudaMalloc(&d_csrValA, edge*sizeof(typeVal));
    //cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    //cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    //cudaMalloc(&d_cooRowIndA, edge*sizeof(int));
    //cudaMalloc(&d_cooColIndA, edge*sizeof(int));
    cudaMalloc(&d_cscValA, edge*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // 5b GPU memory for matrices B and C
    cudaMalloc(&d_cscValB, edge_B*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndB, edge_B*sizeof(int));
    cudaMalloc(&d_cscColPtrB, (m_B+1)*sizeof(int));
    cudaMalloc(&d_cscValC, edge_C*sizeof(typeVal)); // Allocate C in mXm
    cudaMalloc(&d_cscRowIndC, edge_C*sizeof(int));
    cudaMalloc(&d_cscColPtrC, (m_C+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_cscValA, h_cscValA, (edge)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIndA, h_cscRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscColPtrA, h_cscColPtrA, (m+1)*sizeof(int),cudaMemcpyHostToDevice);

    // 6b Copy data from host to device for matrices B and C
    cudaMemcpy(d_cscValB, h_cscValB, (edge_B)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIndB, h_cscRowIndB, (edge_B)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscColPtrB, h_cscColPtrB, (m+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_cscValC, h_cscValC, (edge_C)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIndC, h_cscRowIndC, (edge_C)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscColPtrC, h_cscColPtrC, (m+1)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. [insert CPU verification code here] 

    // 8. Make GPU timers
    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();

    // 9. Initialize product matrix D
    

    // 10. Print two matrices

    // 11. Run spgemm
    //mXm<typeVal>( edge, m, d_cscValB, d_cscColPtrB, d_cscRowIndB, d_cscValC, h_cscColPtrC, d_cscColPtrC, d_cscRowIndC, d_cscValD, d_cscColPtrD, d_cscRowIndD, *context);
    edge_D = spgemm<typeVal>( edge, m, d_cscValB, d_cscColPtrB, d_cscRowIndB, d_cscValC, d_cscColPtrC, d_cscRowIndC, d_cscValD, d_cscColPtrD, d_cscRowIndD );

    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();

    printf("CSR->CSC finished in %f msec. performed %d iterations\n", elapsed, depth-1);
    printf("GPU BFS finished in %f msec. not including transpose\n", elapsed2);

    cudaMemcpy(h_cscRowIndA, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    print_array(h_cscRowIndA, m);

    // 11. Compare with CPU BFS for errors
    cudaMemcpy(h_bfsResult,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    verify( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);
    h_cscValC = (typeVal*)malloc(edge_C*sizeof(typeVal));
    h_cscRowIndC = (int*)malloc(edge_C*sizeof(int));
    cudaMemcpy(h_cscRowIndC, d_cscRowIndC, edge_C*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtrC, d_cscColPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValC, d_cscValC, edge_C*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Matrix C: %dx%d with %d nnz\n", m, m, edge_C);
    print_matrix( h_cscValC, h_cscColPtrC, h_cscRowIndC, m );
}

int main(int argc, char**argv) {
    runBfs(argc, argv);
}    
