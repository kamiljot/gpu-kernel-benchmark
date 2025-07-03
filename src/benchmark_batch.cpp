// Batch-mode benchmark runner: runs benchmarks for multiple input sizes and saves results to CSV.

#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <operation> [passes] [--variant <global|shared|float4|all>]\n";
        return 1;
    }

    std::string operation = argv[1];
    int passes = 100; // default
    std::string variant = "all";

    // Parse passes if given
    int arg_idx = 2;
    if (argc > arg_idx) {
        try {
            passes = std::stoi(argv[arg_idx]);
            arg_idx++;
        }
        catch (...) {
            // not a number, skip
        }
    }

    // Parse optional --variant argument
    for (int i = arg_idx; i < argc; ++i) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variant = argv[i + 1];
            i++;
        }
    }

    std::vector<int> sizes = { 1000, 10000, 100000, 1000000, 10000000 };

    for (int N : sizes) {
        for (int pass = 1; pass <= passes; ++pass) {
            std::vector<float> a, b;
            generate_random_input(N, a, b);
            std::vector<float> c(N);

            BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), N, variant);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << result.cpu_time
                << " ms, ";

            if (variant == "global" || variant == "all")
                std::cout << "Global: " << result.gpu_global_time << " ms, ";
            if (variant == "shared" || variant == "all")
                std::cout << "Shared: " << result.gpu_shared_time << " ms, ";
            if (variant == "float4" || variant == "all")
                std::cout << "Float4: " << result.gpu_float4_time << " ms, ";

            std::cout << "\n";

            append_result_to_csv("result.csv", operation, N, result);
        }
    }

    return 0;
}