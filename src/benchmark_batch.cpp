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
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "input_generator.h"
#include "kernel_dispatch.h"

/**
 * @brief Entry point for batch-mode benchmark runner.
 *
 * Command line usage:
 *   benchmark_batch <operation> [passes] [--variant <global|shared|float4|all>]
 *                   [--warmup <N>] [--mode <kernel|e2e>]
 *
 * The program generates random inputs for a set of predefined sizes, dispatches the selected
 * operation to CPU and GPU kernels, measures execution times, validates results against CPU
 * reference implementation, and appends timings to a CSV file.
 *
 * @param argc  Argument count.
 * @param argv  Argument values.
 * @return      Exit code (0 on success, 1 on usage error).
 */
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <operation> [passes] [--variant <global|shared|float4|all>]\n";
        return 1;
    }

    std::string operation = argv[1];
    int passes = 100;
    int warmup = 20;
    std::string variant = "all";
    BenchmarkMode mode = BenchmarkMode::KernelOnly;

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
        }
    }

    for (int i = arg_idx; i < argc; ++i)
    {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc)
        {
            variant = argv[i + 1];
            i++;
        }
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
        {
            warmup = std::stoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            std::string m = argv[i + 1];
            if (m == "kernel") mode = BenchmarkMode::KernelOnly;
            else if (m == "e2e") mode = BenchmarkMode::EndToEnd;
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

            BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), N, variant, warmup, passes, mode);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << result.cpu_time << " ms, ";

            if (variant == "global" || variant == "all") std::cout << "Global: " << result.gpu_global_time << " ms, ";
            if (variant == "shared" || variant == "all") std::cout << "Shared: " << result.gpu_shared_time << " ms, ";
            if (variant == "float4" || variant == "all") std::cout << "Float4: " << result.gpu_float4_time << " ms, ";


            auto check_correctness = [&](const std::vector<float>& ref, const std::vector<float>& out, const std::string& label) {
                bool correct = true;
                for (size_t i = 0; i < ref.size(); ++i) {
                    if (std::abs(out[i] - ref[i]) > 1e-4f) {
                        correct = false;
                        std::cout << "[ERROR] " << label << " mismatch at " << i << ": " << out[i] << " vs " << ref[i] << "\n";
                        break;
                    }
                }
                if (correct) std::cout << "[CORRECT] " << label << " result matches CPU\n";
            };

            std::vector<float> ref(c.size());
            if (operation == "add") {
                for (size_t i = 0; i < c.size(); ++i) ref[i] = a[i] + b[i];
            } else if (operation == "sqrt_log") {
                for (size_t i = 0; i < c.size(); ++i) ref[i] = std::sqrt(a[i]) + std::log(b[i] + 1e-6f);
            } else if (operation == "sin_cos_pow_relu") {
                for (size_t i = 0; i < c.size(); ++i) ref[i] = std::max(0.0f, std::pow(std::sin(a[i]) + std::cos(b[i]), 2.0f));
            }

            if (variant == "global" || variant == "all") check_correctness(ref, c, "Global");
            if (variant == "shared" || variant == "all") check_correctness(ref, c, "Shared");
            if (variant == "float4" || variant == "all") check_correctness(ref, c, "Float4");

            std::cout << "\n";

            append_result_to_csv("benchmarks/result.csv", operation, N, result);
        }
    }

    return 0;
}