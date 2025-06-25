#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>
#include <vector>
#include <string>

/**
 * Entry point for batch benchmarking.
 * Benchmarks a specific operation+variant across multiple trials.
 */
int main() {
    const std::string operation = "sqrt_log";
    const std::string variant = "all";
    const std::string input_path = "input_file";
    const int passes = 100;
    const int N = 10000000;

    std::vector<float> a, b;
    generate_random_input(N, a, b);
    write_input_file(input_path, a, b);

    std::vector<float> c(N);

    for (int i = 0; i < passes; ++i) {
        BenchmarkResult result = dispatch_and_benchmark(operation, variant, input_path, a.data(), b.data(), c.data(), N);

        std::cout << "[N = " << N << ", pass = " << (i + 1)
            << "] CPU: " << result.cpu_time
            << " ms, Global: " << result.gpu_global_time
            << " ms, Shared: " << result.gpu_shared_time
            << " ms, Float4: " << result.gpu_float4_time << " ms\n";

        append_result_to_csv("result.csv", operation, N, result);
    }

    return 0;
}