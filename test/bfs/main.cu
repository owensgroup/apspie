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
#include <string.h>

#include <deque>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <moderngpu.cuh>
#include <mpi.h>

#include <util.cuh>
#include <bfs.cuh>
#include <spmspvBfs.cuh>
#include <testBfs.cpp>

void runBfs(int argc, char**argv) { 
    // Initialize MPI
    MPI_Init( &argc, &argv );
    int direct, rank, size;

    int m, n, nnz;
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);

    // Define what filetype nnz value should be stored
    typedef float typeVal;

    // File i/o
    // 1. Open file from command-line 
    // -source 1
    freopen(argv[1],"r",stdin);
    int source;
    int device;
    float delta;
    bool undirected;
    int multi;
    if( parseArgs( argc, argv, source, device, delta, undirected, multi )==true ) {
        printf( "Usage: test apple.mtx -source 5\n");
        return;
    }
    //cudaSetDevice(device);
    printf("Testing %s from source %d\n", argv[1], source);

    // Ensure that RDMA ENABLED CUDA is set correctly
    direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
    if(direct != 1){
        printf ("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
        exit (EXIT_FAILURE);
    }

    // Get MPI rank and size
    // Test whether number of MPI processes matches number passed into commandline
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    //printf("My rank is %d\n", rank);
    if( size!=multi ) {
        printf( "Assigned node count %d != %d desired node count!\n");
        exit( EXIT_FAILURE );
    }

    // 2. Reads in number of nnzs, number of nodes
    //    Note: Need to double # of nnzs in case of undirected, because this affects
    //          how much to allocate
    //if( rank==0 ) {
    readEdge( m, n, nnz, stdin );
    if( undirected ) 
        nnz=2*nnz;

    // 3. Allocate memory depending on how many nnzs are present
    typeVal *h_csrValA, *h_cooValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA, *h_cooColIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    h_csrValA    = (typeVal*)malloc(nnz*sizeof(typeVal));
    h_csrColIndA = (int*)malloc(nnz*sizeof(int));
    h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    h_cooValA    = (typeVal*)malloc(nnz*sizeof(typeVal));
    h_cooColIndA = (int*)malloc(nnz*sizeof(int));
    h_cooRowIndA = (int*)malloc(nnz*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // 4. Read in graph from .mtx file
    CpuTimer cpu_timerRead;
    CpuTimer cpu_timerMake;
    CpuTimer cpu_timerBuild;
    if( undirected ) {
        if(rank==0)printf("Old nnz #: %d\n", nnz);
        cpu_timerRead.Start();
        readMtx<typeVal>( nnz/2, h_cooColIndA, h_cooRowIndA, h_cooValA );
        cpu_timerRead.Stop();
        cpu_timerMake.Start();
        nnz = makeSymmetric( nnz, h_cooColIndA, h_cooRowIndA, h_cooValA );
        cpu_timerMake.Stop();
        if(rank==0)printf("\nUndirected graph has %d nodes, %d nnzs\n", m, nnz);
    } else {
        readMtx<typeVal>( nnz, h_cooColIndA, h_cooRowIndA, h_cooValA );
        if(rank==0)printf("\nDirected graph has %d nodes, %d nnzs\n", m, nnz);
    }
    cpu_timerBuild.Start();
    buildMatrix<typeVal>( h_csrRowPtrA, h_csrColIndA, h_csrValA, m, nnz, h_cooRowIndA, h_cooColIndA, h_cooValA );
    cpu_timerBuild.Stop();
    float elapsedRead = cpu_timerRead.ElapsedMillis();
    float elapsedMake = cpu_timerMake.ElapsedMillis();
    float elapsedBuild= cpu_timerBuild.ElapsedMillis();
    if(rank==0) {
        printf("readMtx: %f ms\n", elapsedRead);
        printf("makeSym: %f ms\n", elapsedMake);
        printf("buildMat: %f ms\n", elapsedBuild);
    }

    /*print_array( h_cooRowIndA, m );
    print_array( h_cooColIndA, m );
    print_array( h_csrRowPtrA, m );
    print_array( h_csrColIndA, m );*/

    // 5. Allocate GPU memory
    // Multi-GPU:
    //   -Option 1:
    //   m=m/multi+1            for all
    //   
    //   nnz=same as Option 2
    //
    //   -Option 2: (not implemented yet)
    //   m=m/multi              if rank!=multi-1
    //   m=m-(multi-1)*m/multi  else
    //
    //   nnz=h_csrRowIndA[(rank+1)*m]-h_csrRowIndA[rank*m]  if rank!=multi-1
    //   nnz=nnz-h_csrRowIndA[rank*m]                      else
    int new_n, new_nnz;
    if( rank==multi-1 ) {
        new_n = m-rank*(m/multi+1)+1;
        new_nnz = nnz - h_csrRowPtrA[rank*(m/multi+1)];
    } else {
        new_n = m/multi+1;
        new_nnz = h_csrRowPtrA[(rank+1)*new_n]-h_csrRowPtrA[rank*new_n];
    }

    typeVal *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;
    typeVal *d_cscValA;
    int *d_cscRowIndA, *d_cscColPtrA;
    int *d_bfsResult;
    cudaMalloc(&d_bfsResult, new_n*sizeof(int));

    cudaMalloc(&d_csrValA, new_nnz*sizeof(typeVal));
    cudaMalloc(&d_csrRowPtrA, (new_n+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, new_nnz*sizeof(int));
    //cudaMalloc(&d_cooRowIndA, new_nnz*sizeof(int));

    cudaMalloc(&d_cscValA, new_nnz*sizeof(typeVal));
    cudaMalloc(&d_cscRowIndA, new_nnz*sizeof(int));
    cudaMalloc(&d_cscColPtrA, (new_n+1)*sizeof(int));

    // 6. Copy data from host to device
    cudaMemcpy(d_csrValA, &h_csrValA[h_csrRowPtrA[rank*(m/multi+1)]], (new_nnz)*sizeof(typeVal),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, &h_csrColIndA[h_csrRowPtrA[rank*(m/multi+1)]], (new_nnz)*sizeof(int),cudaMemcpyHostToDevice);
    if( rank==multi-1 ) {
        cudaMemcpy(d_csrRowPtrA, &h_csrRowPtrA[rank*new_n], (m-rank*new_n+1)*sizeof(int),cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(d_csrRowPtrA, &h_csrRowPtrA[rank*new_n], (new_n+1)*sizeof(int),cudaMemcpyHostToDevice);
    }

    // Test copy data from device to host
    /*typeVal *h_csrValTest = (typeVal*)malloc(nnz*sizeof(typeVal));
    int *h_csrColIndTest = (int*)malloc(nnz*sizeof(int));
    int *h_csrRowPtrTest = (int*)malloc((m+1)*sizeof(int));
    int *h_rank = (int*)malloc(multi*sizeof(int));
    int *h_displs = (int*)malloc(multi*sizeof(int));
    //for( int i=0; i<multi; i++ ) h_displs[i] = h_csrRowPtrA[i*(m/multi+1)];
    for( int i=0; i<multi; i++ ) h_displs[i] = i*new_n;

    typeVal *d_csrValTest;
    int *d_csrRowPtrTest;
    int *d_csrColIndTest;
    int *d_rank;
    int *d_new_nnz;
    cudaMalloc(&d_new_nnz, sizeof(int));
    cudaMalloc(&d_rank, multi*sizeof(int));
    cudaMalloc(&d_csrValTest, nnz*sizeof(typeVal));
    cudaMalloc(&d_csrRowPtrTest, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndTest, nnz*sizeof(int));
    //cudaMemcpy(d_new_nnz, &new_nnz, sizeof(int), cudaMemcpyHostToDevice);
    printf("%d: %d col, %d nnz\n", rank, new_n, new_nnz);
    //MPI_Barrier(MPI_COMM_WORLD);
    
    //MPI_Gather( &new_nnz, 1, MPI_INT, h_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather( &new_n, 1, MPI_INT, h_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(rank==0)cudaMemcpy(h_rank, d_rank, multi*sizeof(int), cudaMemcpyDeviceToHost);
    if(rank==0)print_array(h_rank, multi);
    for( int i=0; i<multi; i++ ) {
        int valid_rank;
        if( i!=multi-1 ) valid_rank = h_csrRowPtrA[(i+1)*new_n]-h_csrRowPtrA[i*new_n];
        else valid_rank = nnz - h_csrRowPtrA[i*new_n];
        if(rank==0) printf("%d: %d\n", i, valid_rank);
        if( valid_rank != h_rank[i] && rank==0 ) printf("Error %d: %d != %d\n", i, valid_rank, h_rank[i]);
    }

    MPI_Gatherv(d_csrValA, new_nnz, MPI_FLOAT, d_csrValTest, h_rank, h_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(d_csrRowPtrA, new_n, MPI_INT, d_csrRowPtrTest, h_rank, h_displs, MPI_INT, 0, MPI_COMM_WORLD); 
    //if(rank==0)cudaMemcpy(h_csrValTest, d_csrValTest, nnz*sizeof(typeVal),cudaMemcpyDeviceToHost);
    //if(rank==0)verify( nnz, h_csrValTest, h_csrValA );
    if(rank==0)cudaMemcpy(h_csrRowPtrTest, d_csrRowPtrTest, m*sizeof(int),cudaMemcpyDeviceToHost);
    if(rank==0)verify( m, h_csrRowPtrTest, h_csrRowPtrA );
    cudaMemcpy(&h_csrColIndA[h_csrRowPtrA[rank*new_n]], d_csrColIndA, (new_nnz)*sizeof(int),cudaMemcpyDeviceToHost);
    if( rank==multi-1 ) {
        cudaMemcpy(&h_csrRowPtrA[rank*new_n], d_csrRowPtrA, (m-rank*new_n+1)*sizeof(int),cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(&h_csrRowPtrA[rank*new_n], d_csrRowPtrA, (new_n+1)*sizeof(int),cudaMemcpyDeviceToHost);
    }*/

    // 7. Run COO -> CSR kernel
    //coo2csr( d_cooRowIndA, new_nnz, new_n, d_csrRowPtrA );

    // 8. Run BFS on CPU. Need data in CSR form first.
    int depth = 1000;
    depth = bfsCPU( source, m, h_csrRowPtrA, h_csrColIndA, h_bfsResultCPU, depth );

    // 9. Run CSR -> CSC kernel
    //csr2csc<typeVal>( new_n, new_nnz, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA );

    // 10. Run BFS kernel on GPU
    // Experiment 1: Optimized BFS using mXv (no Val array)
    //spmspvBfs( source, nnz, m, h_csrRowPtrA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, *context); 

    // Experiment 2: Optimized BFS using mXv
    //bfsSparse( source, new_nnz, new_n, m, multi, rank, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_bfsResult, depth, *context); 
    // Compare with CPU BFS for errors
    int *h_bfsResultSmall = (int*)malloc(new_n*sizeof(int));
    cudaMemcpy(h_bfsResultSmall,d_bfsResult,m*sizeof(int),cudaMemcpyDeviceToHost);
    int *h_rank = (int*)malloc(multi*sizeof(int));
    int *h_displs = (int*)malloc(multi*sizeof(int));
    for( int i=0; i<multi; i++ ) h_displs[i] = i*new_n;
    MPI_Gather( &new_n, 1, MPI_INT, h_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(h_bfsResultSmall, new_n, MPI_INT, h_bfsResult, h_rank, h_displs, MPI_INT, 0, MPI_COMM_WORLD); 
    verify( m, h_bfsResult, h_bfsResultCPU );
    //print_array(h_bfsResult, m);



    /*cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);

    cudaFree(d_cscValA);
    cudaFree(d_cscRowIndA);
    cudaFree(d_cscColPtrA);
    cudaFree(d_bfsResult);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooValA);
    free(h_cooRowIndA);
    free(h_cooColIndA);
    free(h_bfsResult);
    free(h_bfsResultCPU);*/

    MPI_Finalize();
}

int main(int argc, char**argv) {
    runBfs(argc, argv);
}    
