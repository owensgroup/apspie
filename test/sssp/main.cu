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
#include <nvgraph.h>

#include <moderngpu.cuh>
#include <util.cuh>
#include <sssp.cuh>

#include <testBfs.cpp>
#include <testSssp.cpp>
#include <string.h>

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
    bool undirected = false;
    if( parseArgs( argc, argv, source, device, delta, undirected )==true ) {
        printf( "Usage: test apple.mtx -source 5\n");
        return;
    }
    //cudaSetDevice(device);
    printf("Testing %s from source %d\n", argv[1], source);
    
    // 2. Reads in number of edges, number of nodes
    //    Note: Need to double # of edges in case of undirected, because this affects
    //          how much to allocate
    readEdge( m, n, edge, stdin );
    if( undirected ) 
        edge=2*edge;

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_csrValA, *h_cooValA, *h_cscValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA, *h_cooColIndA;
    float *h_ssspResult, *h_ssspResultCPU;
    int *h_cscColPtrA, *h_cscRowIndA;

    h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_cscValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cscRowIndA = (int*)malloc(edge*sizeof(int));
    h_cscColPtrA = (int*)malloc((m+1)*sizeof(int));
    h_cooValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cooColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_ssspResult = (float*)malloc((m)*sizeof(float));
    h_ssspResultCPU = (float*)malloc((m)*sizeof(float));

    // 4. Read in graph from .mtx file
    CpuTimer cpu_timerRead;
    CpuTimer cpu_timerMake;
    CpuTimer cpu_timerBuild;
    if( undirected ) {
        printf("Old edge #: %d\n", edge);
        cpu_timerRead.Start();
        readMtx<typeVal>( edge/2, h_cooColIndA, h_cooRowIndA, h_cooValA );
        cpu_timerRead.Stop();
        cpu_timerMake.Start();
        edge = makeSymmetric( edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
        cpu_timerMake.Stop();
        printf("\nUndirected graph has %d nodes, %d edges\n", m, edge);
    } else {
        readMtx<typeVal>( edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
        printf("\nDirected graph has %d nodes, %d edges\n", m, edge);
    }
    cpu_timerBuild.Start();
    buildMatrix<typeVal>( h_csrRowPtrA, h_csrColIndA, h_csrValA, m, edge, h_cooRowIndA, h_cooColIndA, h_cooValA );
    cpu_timerBuild.Stop();
    float elapsedRead = cpu_timerRead.ElapsedMillis();
    float elapsedMake = cpu_timerMake.ElapsedMillis();
    float elapsedBuild= cpu_timerBuild.ElapsedMillis();
    printf("readMtx: %f ms\n", elapsedRead);
    printf("makeSym: %f ms\n", elapsedMake);
    printf("buildMat: %f ms\n", elapsedBuild);

    /*print_array( h_cooRowIndA, m );
    print_array( h_cooColIndA, m );
    print_array( h_csrRowPtrA, m );
    print_array( h_csrColIndA, m );*/

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
    //cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    cudaMalloc(&d_cscValA, edge*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (m+1)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. Run COO -> CSR kernel
    nvgraphStatus_t status;
    //coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // 8. Run SSSP on CPU. Need data in CSR form first.
    //cudaMemcpy(h_cooRowIndA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    //verify( m, h_cooRowIndA, h_csrRowPtrA );
    //cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    int depth = 1000;
    ssspCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_ssspResultCPU, depth );
    //ssspBoost( source, m, edge, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_ssspResultCPU, depth );
    //depth = bfsCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_ssspResult, depth );

    // 9. Run CSR -> CSC kernel
    csr2csc<typeVal>( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    cudaMemcpy(h_cscColPtrA, d_cscColPtrA, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscRowIndA, d_cscRowIndA, edge*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValA, d_cscValA, edge*sizeof(typeVal), cudaMemcpyDeviceToHost);

    // 10. Run SSSP kernel on GPU
    // Experiment 1: Optimized SSSP using mXv (no Val array)
    //spmspvBfs( source, edge, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_ssspResult, depth, *context); 

    // Experiment 2: Optimized SSSP using mXv
    //nvgraphSSSP( source, edge, m, h_cscValA, h_cscColPtrA, h_cscRowIndA, h_ssspResult );
    nvgraphSSSP( source, edge, m, h_csrValA, h_csrRowPtrA, h_csrColIndA, h_ssspResult );
    //sssp( source, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_ssspResult, depth, *context); 
    // Compare with CPU SSSP for errors
    //cudaMemcpy(h_ssspResult,d_ssspResult,m*sizeof(float),cudaMemcpyDeviceToHost);
    verify( m, h_ssspResult, h_ssspResultCPU );
    //print_array(h_ssspResult, m);

    // Compare with SpMV for errors
    //bfs( 0, edge, m, d_cscColPtrA, d_cscRowIndA, d_ssspResult, depth, *context);
    //cudaMemcpy(h_ssspResult,d_ssspResult,m*sizeof(float),cudaMemcpyDeviceToHost);
    //verify( m, h_ssspResult, h_ssspResultCPU );
    //print_array(h_ssspResult, m);
    
    /*cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);

    cudaFree(d_cscValA);
    cudaFree(d_cscRowIndA);
    cudaFree(d_cscColPtrA);
    cudaFree(d_ssspResult);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooValA);
    free(h_cooRowIndA);
    free(h_cooColIndA);
    free(h_ssspResult);
    free(h_ssspResultCPU);*/
}

int main(int argc, char**argv) {
    runSssp(argc, argv);
}    
