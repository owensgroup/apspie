// Puts everything together
// For now, just run V times.
// Optimizations: 
// -come up with good stopping criteria [done]
// -start from i=1 [done]
// -test whether float really are faster than ints
// -distributed idea
// -change nthread [done - doesn't work]
 
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <deque>
#include <cusparse.h>
#include <moderngpu.cuh>

#include <util.cuh>
#include <spgemm.cuh>
#include <testBfs.cpp>
#include <string.h>
#include <fstream>
#include <matrix.hpp>
#include <matrix.cpp>

// Counts number of nnz in 1D partition
void histogramHorz( const int *h_cscColPtrA, const int *h_cscRowIndA, const int m, const int part_size )
{
	std::ofstream outf;
	outf.open("histogramA.csv", std::ofstream::out | std::ofstream::app);
	printf("Partition size: %d\n", part_size);
	outf << "New matrix\n";
	int curr_ind = 0;
	int last_ind = 0;
	for( int i=part_size; i<=m-part_size; i+=part_size )
	{
		last_ind = curr_ind;
		curr_ind = h_cscColPtrA[i];
		outf << i/part_size-1 << " " << curr_ind-last_ind << "\n";
	}
}

// Counts number of nnz in 1D partition
void histogramVert( const int *h_cscColPtrA, const int *h_cscRowIndA, const int m, const int part_size )
{
	std::ofstream outf;
	outf.open("histogramB.csv", std::ofstream::out | std::ofstream::app);
	printf("Partition size: %d\n", part_size);
	outf << "New matrix\n";
	int curr_ind = 0;
	int last_ind = 0;
	for( int i=part_size; i<=m-part_size; i+=part_size )
	{
		last_ind = curr_ind;
		curr_ind = h_cscColPtrA[i];
		outf << i/part_size-1 << " " << curr_ind-last_ind << "\n";
	}
}

// Counts number of nnz in 2D partition
void histogramBlock( const int *h_cscColPtrA, const int *h_cscRowIndA, const int m, const int part_size )
{
	std::ofstream outf, outf2;
	outf2.open("histogramC.csv", std::ofstream::out | std::ofstream::app);
	outf2 << "New matrix\n";

	printf("Partition size: %d\n", part_size);
	int curr_ind = 0;
	int last_ind = 0;

    int block_size = (m+part_size-1)/part_size;
	int *block = (int*) malloc( block_size*block_size*sizeof(int));
	printf("Block size: %d\n", block_size);
	for( int i=0; i<block_size*block_size; i++ )
		block[i] = 0;

	for( int i=part_size; i<=m-part_size; i+=part_size )
	{
		last_ind = curr_ind;
		curr_ind = h_cscColPtrA[i];
		for( int j=last_ind; j< curr_ind; j++ ) 
		{
			//printf("%d ", (i/part_size-1)*block_size+h_cscRowIndA[j]/part_size ); 
			block[(i/part_size-1)*block_size+h_cscRowIndA[j]/part_size]++;
		}
	}

	for( int i=0; i<block_size*block_size; i++ )
	{
		if( block[i]!=0 )
			outf2 << i/block_size << " " << i%block_size << " " << block[i] << "\n";
	}
}

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
                //printf("%d %d %d\n", i, j, count);
                h_cscRowIndB[count] = h_cscRowIndA[j];
                h_cscValB[count] = h_cscValA[j];
                count++;
            }
        }
        h_cscColPtrB[i+1] = count;
    }
}

long long squareDegree( const int m, int *h_cscColPtrA ) {
    long long sum = 0;
    long long deg = 0;
    for( int i=0; i<m; i++ ) {
        deg = h_cscColPtrA[i+1] - h_cscColPtrA[i];
        sum += deg*deg;
    }
    return sum;
}

int maxDegree( const int m, int *h_cscColPtrA ) {
    int max = 0;
    int deg = 0;
    for( int i=0; i<m; i++ ) {
        deg = h_cscColPtrA[i+1] - h_cscColPtrA[i];
        if( deg > max )
            max = deg;
    }
    return max;
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
	bool weighted = false;
    if( parseArgs( argc, argv, source, device, delta, undirected )==true ) {
        printf( "Usage: test apple.mtx -source 5\n");
        return;
    }
    //cudaSetDevice(device);
    printf("Testing %s from source %d\n", argv[1], source);
    
    // 2. Reads in number of edges, number of nodes
    readEdge( m, n, edge, stdin );
    printf("Graph has %d nodes, %d edges\n", m, edge);
    if( undirected )
      edge=2*edge;

    // 3. Allocate memory depending on how many edges are present
    typeVal *h_cooValA;
    //int *h_csrRowPtrA, *h_csrColIndA;
    int *h_cooRowIndA, *h_cooColIndA;
    int *h_bfsResult, *h_bfsResultCPU;

    //h_csrValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cooValA    = (typeVal*)malloc(edge*sizeof(typeVal));
    //h_csrRowPtrA = (int*)malloc((m+1)*sizeof(int));
    //h_csrColIndA = (int*)malloc(edge*sizeof(int));
    h_cooRowIndA = (int*)malloc(edge*sizeof(int));
    h_cooColIndA = (int*)malloc(edge*sizeof(int));
    h_bfsResult = (int*)malloc((m)*sizeof(int));
    h_bfsResultCPU = (int*)malloc((m)*sizeof(int));

    // 4. Read in graph from .mtx file
    // We are actually doing readMtx<typeVal>( edge, h_cooColIndA, h_cscRowIndA, h_cscValA );
    CpuTimer cpu_timerRead;
    CpuTimer cpu_timerMake;
    CpuTimer cpu_timerBuild;
	d_matrix A;
	d_matrix B; //Same as A
	d_matrix C;
	d_matrix D; //Used for counting vertical slab nnzs and for my spgemm mult
	matrix_new( &A, m, m );
	matrix_new( &B, m, m );

    cpu_timerRead.Start();
	if( undirected )
    	weighted = readMtx<typeVal>( edge/2, h_cooColIndA, h_cooRowIndA, h_cooValA );
	else
    	weighted = readMtx<typeVal>( edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
    cpu_timerRead.Stop();
    if( !weighted )
		buildVal( edge, h_cooValA );
    cpu_timerMake.Start();
    if( undirected ) {
		edge = makeSymmetric( edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
    	printf("\nUndirected graph has %d nodes, %d edges\n", m, edge);
	}
    //print_matrixCOO( h_cooValA, h_cooRowIndA, h_cooColIndA, m, edge );
    cpu_timerMake.Stop();
    cpu_timerBuild.Start();
    // This function reads CSR or CSC by swapping h_cooColIndA and h_cooRowIndA
    if( undirected )
		buildMatrix<typeVal>( &A, edge, h_cooRowIndA, h_cooColIndA, h_cooValA );
	else
    	buildMatrix<typeVal>( &A, edge, h_cooColIndA, h_cooRowIndA, h_cooValA );
    cpu_timerBuild.Stop();
    float elapsedRead = cpu_timerRead.ElapsedMillis();
    float elapsedMake = cpu_timerMake.ElapsedMillis();
    float elapsedBuild= cpu_timerBuild.ElapsedMillis();
    printf("readMtx: %f ms\n", elapsedRead);
    printf("makeSym: %f ms\n", elapsedMake);
    printf("buildMat: %f ms\n", elapsedBuild);

    // 4b. Count diagonal
    int diag = countDiag( edge, h_cooRowIndA, h_cooColIndA ); 
    printf("Number of elements on diagonal: %d\n", diag);
    printf("Number of elements on L: %d\n", A.nnz);
    printf("The max degree is: %d\n", maxDegree(m, A.h_cscColPtr));
    printf("Square degree sum is: %lld\n", squareDegree(m, A.h_cscColPtr));

    //buildLower( m, edge_B, h_cscColPtrA, h_cscRowIndA, h_cscValA, h_cscColPtrB, h_cscRowIndB, h_cscValB );
    print_matrix( &A, A.m, true );

    // 5. Allocate GPU memory
    //typeVal *d_csrValA;
    //int *d_csrRowPtrA, *d_csrColIndA;
    int *d_cooColIndA;
    int *d_bfsResult;
    cudaMalloc(&d_bfsResult, m*sizeof(int));

    // 6. Copy data from host to device
	matrix_copy( &B, &A );

    // 7. [insert CPU verification code here] 

    // 8. Make GPU timers
    GpuTimer gpu_timer;
    float elapsed = 0.0f;

    // 9. Run spgemm
    // Must be UxL because we are using CSC matrices rather than specified CSR
    // input required by cuSPARSE
    int NT = 512;
    int NB = (m+NT-1)/NT;
    gpu_timer.Start();

    /*
    //edge_D = mXm<typeVal>( edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_cscValA, h_cscColPtrA, d_cscColPtrA, d_cscRowIndA, d_cscValD, d_cscColPtrD, d_cscRowIndD, *context);
    edge_C = spgemm<typeVal>( edge, m, d_cscValA, d_cscColPtrA, d_cscRowIndA, d_cscValB, d_cscColPtrB, d_cscRowIndB, d_cscValC, d_cscColPtrC, d_cscRowIndC );
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("spgemm finished in %f msec.\n", elapsed);

    // 10. Allocate memory for C on Host
	typeVal *h_cscValC;
    int *h_cscRowIndC, *h_cscColPtrC;
    h_cscValC = (typeVal*)malloc(edge_C*sizeof(typeVal));
    h_cscRowIndC = (int*)malloc(edge_C*sizeof(int));
    h_cscColPtrC = (int*)malloc((m_C+1)*sizeof(int));

	// Statistics:
	// MEMORY = 128000 (L2), 1000 (L1)
	typeVal *h_cscValD;
	int *h_cscRowIndD, *h_cscColPtrD;
    h_cscValD = (typeVal*)malloc(edge*sizeof(typeVal));
    h_cscRowIndD = (int*)malloc(edge*sizeof(int));
    h_cscColPtrD = (int*)malloc((m+1)*sizeof(int));
    buildMatrix<typeVal>( h_cscColPtrD, h_cscRowIndD, h_cscValD, m, edge, h_cooRowIndA, h_cooColIndA, h_cooValA );
	float AGGRO_FACTOR = 0.5;  // a value in (0-1] that describes how close to 
								// shared mem threshold
	float k_A = (float)edge/m;
	float MEMORY = 128000.0;
	float TARGET_PART_SIZE = AGGRO_FACTOR*MEMORY/k_A;
	float TARGET_PART_NUM = edge/MEMORY/AGGRO_FACTOR;

	printf("Mem: %f; Size: %f; Num: %f\n", MEMORY, (int)TARGET_PART_SIZE, TARGET_PART_NUM);

    // 11. Compare with CPU BFS for errors
    cudaMemcpy(h_cscRowIndC, d_cscRowIndC, edge_C*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtrC, d_cscColPtrC, (m_C+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscValC, d_cscValC, edge_C*sizeof(typeVal), cudaMemcpyDeviceToHost);
    //printf("Matrix C: %dx%d with %d nnz\n", m, m, edge_C);
    //print_matrix( h_cscValC, h_cscColPtrC, h_cscRowIndC, m );

    histogramHorz( h_cscColPtrA, h_cscRowIndA, m, (int)TARGET_PART_SIZE );
    histogramVert( h_cscColPtrD, h_cscRowIndD, m, (int)TARGET_PART_SIZE );

	histogramBlock( h_cscColPtrC, h_cscRowIndC, m, (int)TARGET_PART_SIZE );*/
}

int main(int argc, char**argv) {
    runBfs(argc, argv);
}    
