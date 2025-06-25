#pragma once

#include <string>
#include <vector>
#include "benchmark_utils.h"

// Dispatches and benchmarks the specified operation and kernel variant
BenchmarkResult dispatch_and_benchmark(
    const std::string& operation,
    const std::string& variant,
    const std::string& input_path,
    const float* a,
    const float* b,
    float* c,
    int N
);
