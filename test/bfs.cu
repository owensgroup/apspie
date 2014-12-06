// Puts everything together

#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusparse.h>

/*void read_mtx( char** filename ) {
    freopen(filename,"r",stdin);
    //freopen("sample.out","w",stdout);

    scanf("%d", &
}*/

void print_array( int* array, int length ) {
    for( int j=0;j<length;j++ ) {
        printf("%d %d\n", array[j], j);
    }
}

int main(int argc, char** argv) {
    int m, n, edge;

    //for( int i=1; i<argc; i++ ) {
    //bfs(argv[i]);
    freopen(argv[1],"r",stdin);
    //freopen("sample.out","w",stdout);

    scanf("%d %d %d", &m, &n, &edge);
    
    // Allocate memory depending on how many edges are present
    float *h_csrValA;
    int *h_csrRowPtrA, *h_csrColIndA, *h_cooRowIndA;

    h_csrValA    = (float *)malloc(edge*sizeof(float));
    h_csrRowPtrA = (int *)malloc((m+1)*sizeof(int));
    h_csrColIndA = (int *)malloc(edge*sizeof(int));
    h_cooRowIndA = (int *)malloc(edge*sizeof(int));

    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_csrColIndA[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowIndA[j]);
        h_csrValA[j]=1.0;
        printf("%d %d %d\n", h_cooRowIndA[j], h_csrColIndA[j], j);
    }

    // Allocate GPU memory
    float *d_csrValA;
    int *d_csrRowPtrA, *d_csrColIndA, *d_cooRowIndA;

    cudaMalloc(&d_csrValA, edge*sizeof(float));
    cudaMalloc(&d_csrRowPtrA, (m+1)*sizeof(int));
    cudaMalloc(&d_csrColIndA, edge*sizeof(int));
    cudaMalloc(&d_cooRowIndA, edge*sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_csrValA, h_csrValA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooRowIndA, h_cooRowIndA, (edge)*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(h_cooRowIndA, d_cooRowIndA, (edge)*sizeof(int),cudaMemcpyDeviceToHost);
    //print_array(h_cooRowIndA,edge);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Convert from COO -> CSR
    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            printf("COO -> CSR conversion successful!\n");
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("Error: Library not initialized.\n");
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("Error: Invalid value for idxbase.\n");
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("Error: Failed to launch GPU.\n");
    }

    // BFS

    // Copy data back to host
    cudaMemcpy(h_csrRowPtrA,d_csrRowPtrA,(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    print_array(h_csrRowPtrA,m+1);

    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_cooRowIndA);

    free(h_csrValA);
    free(h_csrRowPtrA);
    free(h_csrColIndA);
    free(h_cooRowIndA);
}
