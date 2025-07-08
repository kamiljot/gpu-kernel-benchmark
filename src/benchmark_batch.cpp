/**
 * @file    benchmark_batch.cpp
 * @brief   Batch-mode benchmark runner: runs benchmarks for multiple input sizes and saves results to CSV.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * This program runs selected benchmark operations (CPU and various GPU kernel variants) for a set of input sizes,
 * logs timings, and saves results to a CSV file.
 */

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "input_generator.h"
#include "kernel_dispatch.h"

/**
 * @brief Entry point for the batch-mode benchmark runner.
 *
 * Usage:
 *   benchmark_batch <operation> [passes] [--variant <global|shared|float4|all>]
 *
 * - Runs benchmarks for a range of input sizes, a specified number of passes, and kernel variants.
 * - Results are printed to stdout and saved to a CSV file.
 *
 * @param[in]  argc  Argument count.
 * @param[in]  argv  Argument values.
 * @return           Exit code (0 on success).
 */
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <operation> [passes] [--variant <global|shared|float4|all>]\n";
        return 1;
    }

    std::string operation = argv[1];
    int passes = 100;  // default
    std::string variant = "all";

    // Parse passes if given
    int arg_idx = 2;
    if (argc > arg_idx)
    {
        try
        {
            passes = std::stoi(argv[arg_idx]);
            arg_idx++;
        }
        catch (...)
        {
            // not a number, skip
        }
    }

    // Parse optional --variant argument
    for (int i = arg_idx; i < argc; ++i)
    {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc)
        {
            variant = argv[i + 1];
            i++;
        }
    }

    std::vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};

    for (int N : sizes)
    {
        for (int pass = 1; pass <= passes; ++pass)
        {
            std::vector<float> a, b;
            generate_random_input(N, a, b);
            std::vector<float> c(N);

            BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), N, variant);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << result.cpu_time << " ms, ";

            if (variant == "global" || variant == "all") std::cout << "Global: " << result.gpu_global_time << " ms, ";
            if (variant == "shared" || variant == "all") std::cout << "Shared: " << result.gpu_shared_time << " ms, ";
            if (variant == "float4" || variant == "all") std::cout << "Float4: " << result.gpu_float4_time << " ms, ";

            std::cout << "\n";

            append_result_to_csv("result.csv", operation, N, result);
        }
    }

    return 0;
}