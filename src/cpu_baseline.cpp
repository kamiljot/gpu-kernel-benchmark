#include "cpu_baseline.h"
#include <cmath>
#include <chrono>

float run_cpu_add(const float* a, const float* b, float* c, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        c[i] = std::sqrt(a[i]) + std::log(b[i] + 1e-6f);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}
