// Provides utility functions

#include <ctime>
#include <iostream>
#include <sys/resource.h>

template<typename T>
void print_end_interesting( T *array, int length ) {
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
void print_end( T *array, int length ) {
    int start = length > 10 ? length-10 : 0;
    for( int j=start;j<length;j++ ) {
        std::cout << array[j] << " ";
    }
    std::cout << "\n";
}

template<typename T>
void print_array( T *array, int length ) {
    if( length>40 ) length=40;
    for( int j=0;j<length;j++ ) {
        std::cout << "[" << j << "]:" << array[j] << " ";
    }
    std::cout << "\n";
}

struct CpuTimer {
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
int CompareResults(T* computed, T* reference, int len, bool verbose = true)
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

// Verify the result
void verify( const int m, const int *h_bfsResult, const int *h_bfsResultCPU ){
    if (h_bfsResultCPU != NULL) {
        printf("Label Validity: ");
        int error_num = CompareResults(h_bfsResult, h_bfsResultCPU, m, true);
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

