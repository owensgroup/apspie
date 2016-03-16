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
#include <cusparse.h>
#include <moderngpu.cuh>

#include <util.cuh>
#include <sssp.cuh>

#include <string.h>
#include <testBfs.cpp>
#include <testSssp.cpp>

void runSssp(int argc, char**argv) { 
    int m, n, edge;
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);

    // Define what filetype edge value should be stored
    typedef float Value;

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
    Value *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;
    float *h_ssspResult, *h_ssspResultCPU;

    h_csrValA    = (Value*)malloc(edge*sizeof(Value));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_ssspResult = (float*)malloc((m)*sizeof(float));
    h_ssspResultCPU = (float*)malloc((m)*sizeof(float));

    // 4. Read in graph from .mtx file
    readMtx<Value>( edge, h_csrColIndA, h_cooRowIndA, h_csrValA );
    print_array( h_cooRowIndA, m );

    // 5. Allocate GPU memory
    Value *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    Value *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;
    float *d_ssspResult;
    cudaMalloc(&d_ssspResult, m*sizeof(float));

    cudaMalloc(&d_csrValA, edge*sizeof(Value));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    cudaMalloc(&d_cscValA, edge*sizeof(Value));
    cudaMalloc(&d_cscRowIndA, edge*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (m+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(Value),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);

    // 7. Run COO -> CSR kernel
    coo2csr( d_cooRowIndA, edge, m, d_csrRowPtrA );

    // 8. Run SSSP on CPU. Need data in CSR form first.
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    int depth = 1000;
    ssspCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_ssspResultCPU, depth );
    print_end_interesting(h_ssspResultCPU, m);

    // Verify SSSP CPU with BFS CPU.
    depth = bfsCPU<float>( source, m, h_csrRowPtrA, h_csrColIndA, h_ssspResult, 1000);
    //ssspBoost( source, m, edge, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_ssspResult, 1000);
    //verify<float>( m, h_ssspResultCPU, h_ssspResult );

    // Make two GPU timers
    GpuTimer gpu_timer;
    GpuTimer gpu_timer2;
    float elapsed = 0.0f;
    float elapsed2 = 0.0f;
    gpu_timer.Start();

    // 9. Run CSR -> CSC kernel
    csr2csc<Value>( m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );
    gpu_timer.Stop();
    gpu_timer2.Start();

    // 10. Run SSSP kernel on GPU
    //sssp<Value>( source, edge, m, d_csrValA, d_cscColPtrA, d_cscRowIndA, d_ssspResult, depth, *context );
    sssp<Value>( source, edge, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_ssspResult, depth, *context );

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
