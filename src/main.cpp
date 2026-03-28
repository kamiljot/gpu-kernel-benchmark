/**
 * @file    main.cpp
 * @brief   Main program: loads/generates input, runs selected operation, logs results.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Entry point for running a selected math operation (CPU/GPU) on loaded or generated input,
 * printing and saving results to CSV.
 */

#include <cstring>
#include <iostream>

#include "benchmark_utils.h"
#include "input_generator.h"
#include "kernel_dispatch.h"

namespace
{
bool is_valid_operation(const std::string& op)
{
    return op == "add" || op == "sqrt_log" || op == "sin_cos_pow_relu";
}

bool is_valid_variant(const std::string& v)
{
    return v == "global" || v == "shared" || v == "float4" || v == "all";
}

void print_usage(const char* exe)
{
    std::cout << "Usage:\n"
              << "  " << exe << " <operation> <input_file> [--variant <global|shared|float4|all>]\n"
              << "       [--warmup <N>] [--passes <N>] [--mode <kernel|e2e>] [--help]\n";
}
} // namespace

/**
 * @brief Entry point for single-batch benchmark program.
 *
 * @param argc  Argument count.
 * @param argv  Argument values.
 * @return      Exit code (0 on success).
 */
int main(int argc, char* argv[])
{
    if (argc < 2 || std::string(argv[1]) == "--help")
    {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string operation = argv[1];
    std::string input_path = (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) ? argv[2] : "input_file.bin";
    std::string variant = "all";
    int warmup = 20;
    int passes = 500;
    BenchmarkMode mode = BenchmarkMode::KernelOnly;

    int start_idx = (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) ? 3 : 2;

    // Parse optional --variant, --warmup, --passes, --mode arguments
    for (int i = start_idx; i < argc; ++i)
    {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc)
        {
            variant = argv[++i];
        }
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
        {
            warmup = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc)
        {
            passes = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            std::string m = argv[++i];
            if (m == "kernel") mode = BenchmarkMode::KernelOnly;
            else if (m == "e2e") mode = BenchmarkMode::EndToEnd;
            else
            {
                std::cerr << "Invalid --mode value: " << m << " (expected: kernel|e2e)\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!is_valid_operation(operation))
    {
        std::cerr << "Invalid operation: " << operation
                  << " (expected: add|sqrt_log|sin_cos_pow_relu)\n";
        return 1;
    }

    if (!is_valid_variant(variant))
    {
        std::cerr << "Invalid variant: " << variant
                  << " (expected: global|shared|float4|all)\n";
        return 1;
    }

    if (passes <= 0 || warmup < 0)
    {
        std::cerr << "Invalid numeric values: passes must be > 0 and warmup >= 0\n";
        return 1;
    }

    std::vector<float> a, b;
    if (!read_input_file(input_path, a, b))
    {
        std::cout << "Input file not found. Generating random data...\n";
        const int N = 1000000;
        generate_random_input(N, a, b);
        write_input_file(input_path, a, b);
    }

    std::vector<float> c(a.size());

    std::cout << "Starting dispatch_and_benchmark...\n";
    BenchmarkResult result =
        dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), static_cast<int>(a.size()), variant, warmup,
                                passes, mode);
    std::cout << "Finished dispatch_and_benchmark.\n";

    for (int i = 0; i < 5 && i < static_cast<int>(c.size()); ++i)
    {
        std::cout << "c[" << i << "] = " << c[i] << "\n";
    }

    std::cout << "[N = " << a.size() << "] CPU: " << result.cpu_time << " ms\n";

    if (variant == "global" || variant == "all") std::cout << "Global: " << result.gpu_global_time << " ms\n";
    if (variant == "shared" || variant == "all") std::cout << "Shared: " << result.gpu_shared_time << " ms\n";
    if (variant == "float4" || variant == "all") std::cout << "Float4: " << result.gpu_float4_time << " ms\n";

    const CsvRunMetadata metadata = make_csv_run_metadata(variant, mode, warmup, passes);
    append_result_to_csv("benchmarks/result.csv", operation, static_cast<int>(a.size()), result, metadata);
    return 0;
}