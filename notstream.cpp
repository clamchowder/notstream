#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>;
#include <sys\timeb.h>
#include <immintrin.h>
#include <omp.h>

#define ARR_SIZE_K 32768
#define DATA_G 512

int main(int argc, char *argv[])
{
    struct timeb start, end;
    int64_t time_diff_ms, element_count, arr_size, arr_size_k, num_threads, i, data_g, iterations, iter;
    int64_t* A, * B, * C;
    float bw;

    num_threads = omp_get_num_threads();

    if (argc < 3)
    {
        printf("Usage: [array size in K] [data in G] [thread count]\nUsing %d K for each array and %lld threads, aiming for %d G of data\n", ARR_SIZE_K, num_threads, DATA_G);
        arr_size_k = ARR_SIZE_K;
        data_g = DATA_G;
    }
    else
    {
        arr_size_k = atoi(argv[1]);
        data_g = atoi(argv[2]);
        num_threads = atoi(argv[3]);
        printf("Using %lld K for each array and %lld threads, targeting %lld G of total data transferred\n", arr_size_k, num_threads, data_g);
    }

    element_count = 1024 * arr_size_k / sizeof(int64_t);

    // make element count divisible by 4 so we can use 256-bit ops cleanly
    if (element_count % 4 != 0) element_count += 4 - (element_count % 4);
    arr_size = element_count * sizeof(int64_t);
    iterations = 1024 * 1024 * data_g / (arr_size_k * 3);
    if (iterations == 0) iterations = 1;

    printf("%lld elements, %lld iterations\n", element_count, iterations);

    A = (int64_t*)malloc(arr_size);
    B = (int64_t*)malloc(arr_size);
    C = (int64_t*)malloc(arr_size);

    printf("Initializing arrays\n");
    #pragma omp parallel for
    for (i = 0; i < element_count; i++)
    {
        A[i] = i;
        B[i] = i + 1;
        C[i] = 0;
    }

    omp_set_num_threads(num_threads);

    printf("Running...\n");
    ftime(&start);
    for (iter = 0; iter < iterations; iter++)
        #pragma omp parallel for
        for (i = 0; i < element_count; i += 4) 
        {
            __m256i a = _mm256_load_si256((__m256i*)(A + i));
            __m256i b = _mm256_load_si256((__m256i*)(B + i));
            __m256i c = _mm256_add_epi64(a, b);
            _mm256_store_si256((__m256i*)(C + i), c);
            //C[i] = A[i] + B[i];
        }

    ftime(&end);
    time_diff_ms = 1000 * (end.time - start.time) + (end.millitm - start.millitm);
    bw = iterations * (float)(element_count * sizeof(int64_t) * 3 )/ ((float)time_diff_ms * 1000);
    printf("BW: %f GB/s, in %lld ms\n", bw / 1024, time_diff_ms);

    for (i = 0; i < element_count; i++)
        if (C[i] != A[i] + B[i])
            printf("Mismatch!\n");

    free(A);
    free(B);
    free(C);
    return 0;
}