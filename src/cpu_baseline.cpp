// Reference CPU implementations for benchmarking and validation.

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

float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N) {
    auto relu = [](float x) { return x > 0.0f ? x : 0.0f; };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        float val = std::sin(a[i]) + std::cos(b[i]);
        val = std::pow(val, 2.0f);
        c[i] = relu(val);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diff = end - start;

    return diff.count();
}
