// Provides kernel dispatch and benchmarking utilities for CPU and GPU kernel variants.

#pragma once
#include <string>

// Holds timing results for CPU and various GPU implementations.
struct BenchmarkResult {
    float cpu_time = 0.0f;
    float gpu_global_time = 0.0f;
    float gpu_shared_time = 0.0f;
    float gpu_float4_time = 0.0f;
};

// Dispatches the requested operation (e.g., "add", "sqrt_log"), runs CPU and GPU kernels,
// and returns their execution times (ms) in a BenchmarkResult struct.
// a, b: input arrays
// c: output array
// N: number of elements
BenchmarkResult dispatch_and_benchmark(const std::string& operation,
    const float* a, const float* b, float* c, int N);
