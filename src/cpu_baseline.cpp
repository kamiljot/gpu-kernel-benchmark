#include "cpu_baseline.h"
#include <cmath>
#include <chrono>

float run_cpu_baseline(const std::string& operation, const float* a, const float* b, float* c, int N) {
    auto start = std::chrono::high_resolution_clock::now();

    if (operation == "add") {
        for (int i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    else if (operation == "sqrt_log") {
        for (int i = 0; i < N; ++i) {
            c[i] = std::sqrt(a[i]) + std::log(b[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count();
}

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
    for (int i = 0; i < N; ++i) {
        float value = sinf(a[i]) + cosf(b[i]) + powf(a[i], b[i]);
        c[i] = value > 0 ? value : 0.0f;
    }
    return 0.0f; // Placeholder if timing isn't needed
}