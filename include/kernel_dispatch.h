#pragma once
#include <string>

// Holds timing results for CPU and various GPU implementations
struct BenchmarkResult {
    float cpu_time = 0.0f;
    float gpu_global_time = 0.0f;
    float gpu_shared_time = 0.0f;
    float gpu_float4_time = 0.0f;
};

// Dispatches the appropriate kernel functions based on the operation name
BenchmarkResult dispatch_and_benchmark(const std::string& operation,
    const float* a, const float* b, float* c, int N);