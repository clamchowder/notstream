#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>;
#include <sys\timeb.h>
#include <intrin.h>
#include <immintrin.h>
#include <omp.h>

#define ARR_SIZE_K 32768
#define DATA_G 512

void avx2_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count);
void sse2_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count);
void scalar_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count);

int main(int argc, char *argv[])
{
    struct timeb start, end;
    int64_t time_diff_ms, element_count, arr_size, arr_size_k, num_threads, i, data_g, iterations, iter;
    int64_t* A, * B, * C;
    float bw;
    int cpuid_data[4];
    void(*add_func)(int64_t*, int64_t*, int64_t*, int64_t) = NULL;

    num_threads = omp_get_num_threads();

    if (argc < 4)
    {
        fprintf(stderr, "Usage: [array size in K] [data in G] [thread count] [scalar/sse2/avx2]\nUsing %d K for each array and %lld threads, aiming for %d G of data\n", ARR_SIZE_K, num_threads, DATA_G);
        arr_size_k = ARR_SIZE_K;
        data_g = DATA_G;
    }
    else
    {
        arr_size_k = atoi(argv[1]);
        data_g = atoi(argv[2]);
        num_threads = atoi(argv[3]);
        fprintf(stderr, "Using %lld K for each array and %lld threads, targeting %lld G of total data transferred\n", arr_size_k, num_threads, data_g);
    }

    add_func = scalar_add;
    if (argc == 5)
    {
        if (_strnicmp(argv[4], "sse2", 4) == 0)
        {
            fprintf(stderr, "Using SSE2 add\n");
            add_func = sse2_add;
        }
        else if (_strnicmp(argv[4], "avx2", 4) == 0)
        {
            fprintf(stderr, "Using AVX2 add\n");
            add_func == avx2_add;
        }
        else fprintf(stderr, "Using scalar add\n");
    }
    else 
    {
        // determine whether sse2 or avx2 can be used
        __cpuidex(cpuid_data, 1, 0);
        if (cpuid_data[3] & (1UL << 26)) // EDX bit 26
        {
            fprintf(stderr, "SSE2 supported\n");
            add_func = sse2_add;
        }

        __cpuidex(cpuid_data, 0x7, 0);
        if (cpuid_data[1] & (1UL << 5)) // EBX bit 5
        {
            fprintf(stderr, "AVX2 supported\n");
            add_func = avx2_add;
        }
    }

    element_count = 1024 * arr_size_k / sizeof(int64_t);

    // make element count divisible by 4 so we can use 256-bit ops cleanly
    if (element_count % 4 != 0) element_count += 4 - (element_count % 4);
    arr_size = element_count * sizeof(int64_t);
    iterations = 1024 * 1024 * data_g / (arr_size_k * 3);
    if (iterations == 0) iterations = 1;

    fprintf(stderr, "%lld elements, %lld iterations\n", element_count, iterations);

    A = (int64_t*)malloc(arr_size);
    B = (int64_t*)malloc(arr_size);
    C = (int64_t*)malloc(arr_size);

    // initialize arrays
    #pragma omp parallel for
    for (i = 0; i < element_count; i++)
    {
        A[i] = i;
        B[i] = i + 1;
        C[i] = 0;
    }

    omp_set_num_threads(num_threads);

    fprintf(stderr, "Running...\n");
    ftime(&start);
    for (iter = 0; iter < iterations; iter++)
        add_func(A, B, C, element_count);

    ftime(&end);
    time_diff_ms = 1000 * (end.time - start.time) + (end.millitm - start.millitm);
    bw = iterations * (float)(element_count * sizeof(int64_t) * 3 ) / ((float)time_diff_ms * 1000 * 1024);
    fprintf(stderr, "Add BW: %f GB/s, in %lld ms\n", bw, time_diff_ms);
    printf("%lld, %lld, %f", arr_size_k, num_threads, bw);

    for (i = 0; i < element_count; i++)
        if (C[i] != A[i] + B[i])
            fprintf(stderr, "Mismatch!\n");

    free(A);
    free(B);
    free(C);
    return 0;
}

// Add, using avx2 instructions. Element count must be divisible by 4
void avx2_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count)
{
#pragma omp parallel for
    for (int64_t i = 0; i <= element_count - 4; i += 4)
    {
        __m256i a = _mm256_loadu_si256((__m256i*)(A + i));
        __m256i b = _mm256_loadu_si256((__m256i*)(B + i));
        __m256i c = _mm256_add_epi64(a, b);
        _mm256_storeu_si256((__m256i*)(C + i), c);
    }
}

// Add, using sse2 instructions. Element count must be divisble by 2
void sse2_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count)
{
#pragma omp parallel for
    for (int64_t i = 0; i <= element_count - 2; i += 2)
    {
        __m128i a = _mm_loadu_si128((__m128i*)(A + i));
        __m128i b = _mm_loadu_si128((__m128i*)(B + i));
        __m128i c = _mm_add_epi64(a, b);
        _mm_storeu_si128((__m128i*)(C + i), c);
    }
}

// Add using plain 64-bit integer operations. Or whatever the compiler generates
void scalar_add(int64_t* A, int64_t* B, int64_t* C, int64_t element_count)
{
#pragma omp parallel for
    for (int64_t i = 0; i < element_count; i++)
        C[i] = A[i] + B[i];
}