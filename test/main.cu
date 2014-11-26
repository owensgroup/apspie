// Puts everything together

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#define DIM 3

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}

//Device code for naive matrix multiplication
__device__ void sgemm_helper(const float*A, const float*B, float*C, int z) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x+y*gridDim.x;

    if( z==0 )
        C[offset] = 0;
    C[offset] += B[z+y*gridDim.x] * A[x+z*gridDim.x];
}

__global__ void sgemm(const float*A, const float*B, float*C, const int m, const int n, const int k) {

    int z;

    for( z=0; z<k; z++ ) {
        sgemm_helper(A, B, C, z);
    }

}

//Calculate matrix multiplication
int gpu_sgemm(const float*A, const float*B, float*C, const int m, const int n, const int k) {;

    //Check for invalid input
    if( m<0 || n<0 || k<0 )
        return;

    dim3 grid(m,n);

    sgemm<<<grid,1>>>( A, B, C, m, n, k );
    return;
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//Do matrix calculation before 

int main() {
    clock_t t;

    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    // for simplicity we are going to use square arrays
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
    
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the arrays A and B on GPU with random numbers
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

    // Optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);

    t = clock();
    std::cout << "Calculating...\n";
    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    t = clock() - t;
    std::cout << "It took " << ((float)t)/CLOCKS_PER_SEC << " seconds.\n";

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);

    t = clock();
    std::cout << "Calculating...\n";
    gpu_sgemm(d_A, d_B, d_C, nr_rows_A, nr_cols_B, nr_cols_A);
    t = clock() - t;
    std::cout << "It took " << ((float)t)/CLOCKS_PER_SEC << " seconds.\n";

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);    

    return 0;
}
