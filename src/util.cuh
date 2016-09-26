// Provides utility functions

#include <ctime>
#include <iostream>
#include <sys/resource.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <boost/timer/timer.hpp>
#include "scratch.hpp"

#include <vector>
#include <utility>
#include <algorithm>

template<typename T>
void print_end_interesting( T *array, int length=10 ) {
    int count=0;
    for( int j=length-1;j>=0; j-- ) {
        if( array[(int)j]!=-1) {
            std::cout << "[" << j << "]:" << array[j] << " ";
            count++;
            if( count==9 ) break;
        }
    }
    std::cout << "\n";
}

template<typename T>
void print_end( T *array, int length=10 ) {
    int start = length > 10 ? length-10 : 0;
    for( int j=start;j<length;j++ ) {
        std::cout << array[j] << " ";
    }
    std::cout << "\n";
}

template<typename T>
void print_array( T *array, int length=40 ) {
    if( length>40 ) length=40;
    for( int j=0;j<length;j++ ) {
        std::cout << "[" << j << "]:" << array[j] << " ";
    }
    std::cout << "\n";
}

template<typename T>
void print_matrix( T* h_csrVal, int* h_csrRowPtr, int* h_csrColInd, int length ) {
    //print_array(h_csrRowPtr);
    //print_array(h_csrColInd); 
    //print_array(h_csrVal);
    std::cout << "Matrix:\n";
    if( length>20 ) length=20;
    for( int i=0; i<length; i++ ) {
        int count = h_csrRowPtr[i];
        for( int j=0; j<length; j++ ) {
            if( count>=h_csrRowPtr[i+1] || h_csrColInd[count] != j )
                std::cout << "0 ";
            else {
                std::cout << h_csrVal[count] << " ";
                count++;
            }
        }
        std::cout << std::endl;
    }
}

struct CpuTimer {

#if defined(CLOCK_PROCESS_CPUTIME_ID)

    double start;
    double stop;

    void Start()
    {
        static struct timeval tv;
        static struct timezone tz;
    	gettimeofday(&tv, &tz);
        start = tv.tv_sec + 1.e-6*tv.tv_usec;
    }

    void Stop()
    {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        stop = tv.tv_sec + 1.e-6*tv.tv_usec;
    }

    double ElapsedMillis()
    {
        return 1000*(stop - start);
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec /1000);
    }

#endif
};


/******************************************************************************
 * Helper routines for list construction and validation 
 ******************************************************************************/

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Compares the equivalence of two arrays. If incorrect, print the location
 * of the first incorrect value appears, the incorrect value, and the reference
 * value.
 * \return Zero if two vectors are exactly the same, non-zero if there is any difference.
 *
 */
template <typename T>
int CompareResults(const T* computed, const T* reference, const int len, const bool verbose = true)
{
    int flag = 0;
    for (int i = 0; i < len; i++) {

        if (computed[i] != reference[i] && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            std::cout << computed[i];
            printf(" != ");
            std::cout << reference[i];

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << computed[j];
                    printf(", ");
                }
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << reference[j];
                    printf(", ");
                }
                printf("...]");
            }
            flag += 1;
            //return flag;
        }
        if (computed[i] != reference[i] && flag > 0) flag+=1;
    }
    printf("\n");
    if (flag == 0)
        printf("CORRECT\n");
    return flag;
}

template <>
int CompareResults(const float* computed, const float* reference, const int len, const bool verbose )
{
    int flag = 0;
    for (int i = 0; i < len; i++) {

        if (computed[i] != reference[i] && flag == 0 && !(computed[i]==-1 && reference[i]>1e38) && !(computed[i]>1e38 && reference[i]==-1)) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            std::cout << computed[i];
            printf(" != ");
            std::cout << reference[i];

            printf("\nresult[...");
            for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                std::cout << computed[j];
                printf(", ");
            }
            printf("...]");
            printf("\nreference[...");
            for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                std::cout << reference[j];
                printf(", ");
            }
            printf("...]");
            flag += 1;
            //return flag;
        }
        if (computed[i] != reference[i] && flag > 0 && !(computed[i]==-1 && reference[i]>1e38)) flag+=1;
    }
    printf("\n");
    if (flag == 0)
        printf("CORRECT\n");
    return flag;
}

// Verify the result
template <typename value>
void verify( const int m, const value *h_bfsResult, const value *h_bfsResultCPU ){
    if (h_bfsResultCPU != NULL) {
        printf("Label Validity: ");
        int error_num = CompareResults(h_bfsResult, h_bfsResultCPU, m, true);
        if (error_num > 0) {
            printf("%d errors occurred.\n", error_num);
        }
    }
}

template <typename T>
int uncompareResults(T* computed, T* reference, int len, bool verbose = true)
{
    int flag = 0;
    for (int i = 0; i < len; i++) {

        if (computed[i] + reference[i] != 1 && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
            std::cout << computed[i];
            printf(" == ");
            std::cout << reference[i];

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << computed[j];
                    printf(", ");
                }
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    std::cout << reference[j];
                    printf(", ");
                }
                printf("...]");
            }
            flag += 1;
            //return flag;
        }
        if (computed[i]+reference[i]!=1 && flag > 0) flag+=1;
    }
    printf("\n");
    if (flag == 0)
        printf("CORRECT\n");
    return flag;
}

void unverify( const int m, const int *h_bfsResult, const int *h_bfsResultCPU ) {
    if (h_bfsResultCPU != NULL) {
        printf("Label Validity: ");
        int error_num = uncompareResults(h_bfsResult, h_bfsResultCPU, m, true);
        if (error_num > 0) {
            printf("%d errors occurred.\n", error_num);
        }
    }
}
    
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// This function extracts the number of nodes and edges from input file
void readEdge( int &m, int &n, int &edge, FILE *inputFile ) {
    int c = getchar();
    int old_c = 0;
    while( c!=EOF ) {
        if( (old_c==10 || old_c==0) && c!=37 ) {
            ungetc(c, inputFile);
            break;
        }
        old_c = c;
        c=getchar();
    }
    scanf("%d %d %d", &m, &n, &edge);
}

// This function loads a graph from .mtx input file
template<typename typeVal>
void readMtx( int edge, int *h_cooColInd, int *h_cooRowInd, typeVal *h_cooVal ) {
    bool weighted = true;
    int c;
    int csr_max = 0;
    int csr_current = 0;
    int csr_row = 0;
    int csr_first = 0;

    // Currently checks if there are fewer rows than promised
    // Could add check for edges in diagonal of adjacency matrix
    for( int j=0; j<edge; j++ ) {
        if( scanf("%d", &h_cooColInd[j])==EOF ) {
            printf("Error: not enough rows in mtx file.\n");
            break;
        }
        scanf("%d", &h_cooRowInd[j]);

        if( j==0 ) {
            c=getchar();
        }

        if( c!=32 ) {
            h_cooVal[j]=1.0;
            if( j==0 ) weighted = false;
        } else {
            //std::cin >> h_cooVal[j];
            scanf("%f", &h_cooVal[j]);
        }

        h_cooRowInd[j]--;
        h_cooColInd[j]--;

        //printf("The first row is %d %d\n", h_cooColInd[j], h_cooRowInd[j]);

        // Finds max csr row.
        if( j!=0 ) {
            if( h_cooRowInd[j]==0 ) csr_first++;
            if( h_cooRowInd[j]==h_cooRowInd[j-1] )
                csr_current++;
            else {
                csr_current++;
                //printf("New row: Last row %d elements long\n", csr_current);
                if( csr_current > csr_max ) {
                    csr_max = csr_current;
                    csr_current = 0;
                    csr_row = h_cooRowInd[j-1];
                } else
                    csr_current = 0;
            }
        }
    }
    printf("The biggest row was %d with %d elements.\n", csr_row, csr_max);
    printf("The first row has %d elements.\n", csr_first);
    if( weighted==true ) {
        printf("The graph is weighted. ");
    } else {
        printf("The graph is unweighted.\n");
    }
}

  // This function converts function from COO to CSR representation
  // TODO: -add support for COO->CSC
  //       -add support for int64
  //
  // @tparam[in] <typeVal>   Models value
  //
  // @param[in] numVert
  // @param[in] numEdge
  // @param[in] h_cooRowInd
  // @param[in] h_cooColInd
  // @param[in] h_cooVal
  // @param[out] h_csrRowPtr
  // @param[out] h_csrColInd
  // @param[out] h_csrVal

  template<typename typeVal>
  void buildMatrix( int *h_csrRowPtr,
                    int *h_csrColInd,
                    typeVal *h_csrVal,
                    int numVert,
                    int numEdge,
                    int *h_cooRowInd,     // I
                    int *h_cooColInd,     // J
                    typeVal *h_cooVal ) {

    int temp;
    int row;
    int dest;
    int cumsum = 0;
    //int nnz = numEdge;
    //C.val.resize(nnz);
    //C.rowind.resize(nnz);

    for( int i=0; i<=numVert; i++ )
      h_csrRowPtr[i] = 0;               // Set all rowPtr to 0
    for( int i=0; i<numEdge; i++ )
      h_csrRowPtr[h_cooRowInd[i]]++;                   // Go through all elements to see how many fall into each column
    for( int i=0; i<numVert; i++ ) {                  // Cumulative sum to obtain column pointer array
      temp = h_csrRowPtr[i];
      h_csrRowPtr[i] = cumsum;
      cumsum += temp;
    }
    h_csrRowPtr[numVert] = numEdge;

    for( int i=0; i<numEdge; i++ ) {
      row = h_cooRowInd[i];                         // Store every row index in memory location specified by colptr
      dest = h_csrRowPtr[row];
      h_csrColInd[dest] = h_cooColInd[i];              // Store row index
      h_csrVal[dest] = h_cooVal[i];                 // Store value
      h_csrRowPtr[row]++;                      // Shift destination to right by one
    }
    cumsum = 0;
    for( int i=0; i<=numVert; i++ ) {                 // Undo damage done by moving destination
      temp = h_csrRowPtr[i];
      h_csrRowPtr[i] = cumsum;
      cumsum = temp;
  }}

bool parseArgs( int argc, char**argv, int &source, int &device, float &delta, bool &undirected ) {
    bool error = false;
    source = 0;
    device = 0;
    delta = 0.1;

    for( int i=2; i<argc; i+=2 ) {
       if( strstr(argv[i], "-source") != NULL )
           source = atoi(argv[i+1]);
       else if( strstr(argv[i], "-device") != NULL )
           device = atoi(argv[i+1]);
       else if( strstr(argv[i], "-delta") != NULL )
           delta = atof(argv[i+1]);
       else if( strstr(argv[i], "-undirected") != NULL )
           undirected = true;
    }
    return error;
}

void coo2csr( const int *d_cooRowIndA, const int edge, const int m, int *d_csrRowPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    GpuTimer gpu_timer;
    float elapsed = 0.0f;
    gpu_timer.Start();
    cusparseStatus_t status = cusparseXcoo2csr(handle, d_cooRowIndA, edge, m, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    gpu_timer.Stop();
    elapsed += gpu_timer.ElapsedMillis();
    printf("COO->CSR finished in %f msec. \n", elapsed);

    // Important: destroy handle
    cusparseDestroy(handle);
}

template<typename typeVal>
void csr2csc( const int m, const int edge, const typeVal *d_csrValA, const int *d_csrRowPtrA, const int *d_csrColIndA, typeVal *d_cscValA, int *d_cscRowIndA, int *d_cscColPtrA ) {

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // For CUDA 4.0
    //cusparseStatus_t status = cusparseScsr2csc(handle, m, m, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, 1, CUSPARSE_INDEX_BASE_ZERO);

    // For CUDA 5.0+
    cusparseStatus_t status = cusparseScsr2csc(handle, m, m, edge, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_cscValA, d_cscRowIndA, d_cscColPtrA, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);

    // Important: destroy handle
    cusparseDestroy(handle);
}

// Tuple sort
struct arrayset {
    int *values1;
    int *values2;
    //int *values3;
};

typedef struct pair {
    int key, key2, value;
} Pair;

int cmp(const void *x, const void *y){
    int a = ((const Pair*)x)->key;
    int b = ((const Pair*)y)->key;
    int c = ((const Pair*)x)->key2;
    int d = ((const Pair*)y)->key2;
    if( a==b ) return c < d ? -1 : c > d;
    else return a < b ? -1 : a > b;
}

void custom_sort(struct arrayset *v, size_t size){
    //Pair key[size];
    Pair *key = (Pair *)malloc(size*sizeof(Pair));
    for(int i=0;i<size;++i){
        key[i].key  = v->values1[i];
        key[i].key2 = v->values2[i];
        key[i].value=i;
    }
    qsort(key, size, sizeof(Pair), cmp);
    //int v1[size], v2[size];
    int *v1 = (int*)malloc(size*sizeof(int));
    int *v2 = (int*)malloc(size*sizeof(int));
    memcpy(v1, v->values1, size*sizeof(int));
    memcpy(v2, v->values2, size*sizeof(int));
    //memcpy(v3, v->values3, size*sizeof(int));
    for(int i=0;i<size;++i){
        v->values1[i] = v1[key[i].value];
        v->values2[i] = v2[key[i].value];
    }
    free(key);
    free(v1);
    free(v2);
}

template<typename typeVal>
int makeSymmetric( int edge, int *h_csrColIndA, int *h_cooRowIndA, typeVal *h_randVec ) {

    int realEdge = edge/2;
    
    for( int i=0; i<realEdge; i++ ) {
        h_cooRowIndA[realEdge+i] = h_csrColIndA[i];
        h_csrColIndA[realEdge+i] = h_cooRowIndA[i];
    }

    // Sort
    struct arrayset work = { h_cooRowIndA, h_csrColIndA };
    custom_sort(&work, edge);
    /*std::vector< std::pair<int, int> > pairs;
    for( int i=0; i<edge; i++ )
        pairs.push_back(std::make_pair( h_cooRowIndA[i], h_csrColIndA[i] ));
    std::sort( pairs.begin(), pairs.end() );
    for( int i=0; i<edge; i++ ) {
        h_cooRowIndA[i] = pairs[i].first;
        h_csrColIndA[i] = pairs[i].second;
    }*/

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

//template< typename T >
void allocScratch( d_scratch **d, const int edge, const int m ) {

    *d = (d_scratch *)malloc(sizeof(d_scratch));
    cudaMalloc(&((*d)->d_cscVecInd), edge*sizeof(int));
    cudaMalloc(&((*d)->d_cscSwapInd), edge*sizeof(int));
    cudaMalloc(&((*d)->d_cscVecVal), edge*sizeof(float));
    cudaMalloc(&((*d)->d_cscSwapVal), edge*sizeof(float));
    cudaMalloc(&((*d)->d_cscTempVal), edge*sizeof(float));

    cudaMalloc(&((*d)->d_cscColGood), edge*sizeof(int));
    cudaMalloc(&((*d)->d_cscColBad), m*sizeof(int));
    cudaMalloc(&((*d)->d_cscColDiff), m*sizeof(int));
    cudaMalloc(&((*d)->d_ones), m*sizeof(int));
    cudaMalloc(&((*d)->d_index), m*sizeof(int));
    cudaMalloc(&((*d)->d_randVecInd), m*sizeof(int));

    //Host mallocs
    (*d)->h_cscVecInd = (int*) malloc (edge*sizeof(int));
    (*d)->h_cscVecVal = (float*) malloc (edge*sizeof(float));
    (*d)->h_cscColDiff = (int*) malloc (m*sizeof(int));
    (*d)->h_ones = (int*) malloc (m*sizeof(int));
    (*d)->h_index = (int*) malloc (m*sizeof(int));

    (*d)->h_bfsResult = (int*) malloc (m*sizeof(int));
    (*d)->h_spmvResult = (float*) malloc (m*sizeof(float));
    (*d)->h_bfsValA = (float*) malloc (edge*sizeof(float));
}
