#include <immintrin.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

static inline double mytime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static inline void generate_data(double * T, int size)
{
    for (int i = 0; i < size; i++) {
        T[i] = rand();
    }
}

static inline void kernel(double * T,
                          int size, int intensity, int t)
{
    for (int j = 0; j < intensity; j++) {
        for (int i = 0; i < size; i++)
            T[i] = fma(T[i], (double)t, (double)j);
    }
}

static inline void kernel_loop(double * T,
                               int size, int intensity, int loops)
{
    double s1 = 0.0;
    double s2 = 0.0;
    for (int i = 0; i < loops; i++) {
        generate_data(T, size);
        for (char *C = (char *)T; C < (char *)(T + size); C += 64)
            _mm_clflushopt(C);

        double start = mytime();
        kernel(T, size, intensity, i);
        double end = mytime();
        double duration = end - start;
        s1 += duration;
        s2 += duration * duration;
    }

    double mean = s1 / loops;
    double variance = ((loops * s2) - (s1 * s1)) / (loops * (loops - 1));
    double stddev = sqrt(variance);
    printf("kernel: %f, %f\n", mean, stddev);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
        return -1;

    int size = atoi(argv[1]) / 8;
    int loops = atoi(argv[2]);
    int intensity = atoi(argv[3]);

    double * T = (double*)aligned_alloc(64, sizeof(double[size]));
    kernel_loop(T, size, intensity, loops);
    free(T);

    return 0;
}