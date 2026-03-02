/**
 * @file    benchmark_batch.cpp
 * @brief   Batch-mode benchmark runner with robust CLI validation and per-variant correctness checks.
 * @author  Kamil J.
 * @date    2025-07-07
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "cpu_baseline.h"
#include "input_generator.h"
#include "kernel_dispatch.h"

namespace
{
/**
 * @brief Prints command-line usage.
 * @param exe Executable name (argv[0]).
 */
void print_usage(const char* exe)
{
    std::cout << "Usage:\n"
              << "  " << exe << " <operation> [passes] [--variant <global|shared|float4|all>]\n"
              << "       [--warmup <N>] [--mode <kernel|e2e>] [--help]\n";
}

/**
 * @brief Checks whether operation name is supported.
 * @param op Operation name.
 * @return True if valid, otherwise false.
 */
bool is_valid_operation(const std::string& op)
{
    return op == "add" || op == "sqrt_log" || op == "sin_cos_pow_relu";
}

/**
 * @brief Checks whether GPU variant name is supported.
 * @param v Variant name.
 * @return True if valid, otherwise false.
 */
bool is_valid_variant(const std::string& v)
{
    return v == "global" || v == "shared" || v == "float4" || v == "all";
}

/**
 * @brief Compares output vector against CPU reference.
 * @param ref Reference results.
 * @param out Tested results.
 * @param label Variant label for logging.
 * @return True when all values match tolerance, otherwise false.
 */
bool check_correctness(const std::vector<float>& ref, const std::vector<float>& out, const std::string& label)
{
    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (std::abs(out[i] - ref[i]) > 1e-4f)
        {
            std::cout << "[ERROR] " << label << " mismatch at " << i << ": " << out[i] << " vs " << ref[i] << "\n";
            return false;
        }
    }
    std::cout << "[CORRECT] " << label << " result matches CPU\n";
    return true;
}
} // namespace

/**
 * @brief Program entry point for batch benchmark runs.
 *
 * For each configured input size and pass:
 * - generates input,
 * - runs timing dispatch,
 * - computes CPU reference,
 * - validates each requested GPU variant against reference using dedicated output buffers.
 *
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Exit code (0 on success, non-zero on usage/validation error).
 */
int main(int argc, char* argv[])
{
    if (argc < 2 || std::string(argv[1]) == "--help")
    {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string operation = argv[1];
    int passes = 100;
    int warmup = 20;
    std::string variant = "all";
    BenchmarkMode mode = BenchmarkMode::KernelOnly;

    int arg_idx = 2;
    if (argc > arg_idx && std::string(argv[arg_idx]).rfind("--", 0) != 0)
    {
        try
        {
            passes = std::stoi(argv[arg_idx]);
        }
        catch (...)
        {
            std::cerr << "Invalid passes value: " << argv[arg_idx] << "\n";
            return 1;
        }
        ++arg_idx;
    }

    for (int i = arg_idx; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--variant") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --variant\n";
                return 1;
            }
            variant = argv[++i];
        }
        else if (std::strcmp(argv[i], "--warmup") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --warmup\n";
                return 1;
            }
            try
            {
                warmup = std::stoi(argv[++i]);
            }
            catch (...)
            {
                std::cerr << "Invalid --warmup value\n";
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--mode") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --mode\n";
                return 1;
            }
            const std::string m = argv[++i];
            if (m == "kernel")
            {
                mode = BenchmarkMode::KernelOnly;
            }
            else if (m == "e2e")
            {
                mode = BenchmarkMode::EndToEnd;
            }
            else
            {
                std::cerr << "Invalid --mode value: " << m << " (expected: kernel|e2e)\n";
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--help") == 0)
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
        std::cerr << "Invalid operation: " << operation << "\n";
        return 1;
    }

    if (!is_valid_variant(variant))
    {
        std::cerr << "Invalid variant: " << variant << " (expected: global|shared|float4|all)\n";
        return 1;
    }

    if (passes <= 0 || warmup < 0)
    {
        std::cerr << "Invalid numeric values: passes must be > 0 and warmup >= 0\n";
        return 1;
    }

    const std::vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};

    for (int N : sizes)
    {
        for (int pass = 1; pass <= passes; ++pass)
        {
            std::vector<float> a, b;
            generate_random_input(N, a, b);

            // Timing run
            std::vector<float> timing_out(N);
            BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), timing_out.data(),
                                                            N, variant, warmup, passes, mode);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << result.cpu_time << " ms, ";
            if (variant == "global" || variant == "all") std::cout << "Global: " << result.gpu_global_time << " ms, ";
            if (variant == "shared" || variant == "all") std::cout << "Shared: " << result.gpu_shared_time << " ms, ";
            if (variant == "float4" || variant == "all") std::cout << "Float4: " << result.gpu_float4_time << " ms, ";

            // CPU reference
            std::vector<float> ref(N);
            if (operation == "add")
            {
                run_cpu_add(a.data(), b.data(), ref.data(), N);
            }
            else if (operation == "sqrt_log")
            {
                run_cpu_sqrt_log(a.data(), b.data(), ref.data(), N);
            }
            else
            {
                run_cpu_sin_cos_pow_relu(a.data(), b.data(), ref.data(), N);
            }

            // Per-variant correctness using dedicated buffers
            if (variant == "global" || variant == "all")
            {
                std::vector<float> out_global(N);
                dispatch_and_benchmark(operation, a.data(), b.data(), out_global.data(),
                                       N, "global", warmup, 1, mode);
                check_correctness(ref, out_global, "Global");
            }

            if (variant == "shared" || variant == "all")
            {
                std::vector<float> out_shared(N);
                dispatch_and_benchmark(operation, a.data(), b.data(), out_shared.data(),
                                       N, "shared", warmup, 1, mode);
                check_correctness(ref, out_shared, "Shared");
            }

            if (variant == "float4" || variant == "all")
            {
                std::vector<float> out_float4(N);
                dispatch_and_benchmark(operation, a.data(), b.data(), out_float4.data(),
                                       N, "float4", warmup, 1, mode);
                check_correctness(ref, out_float4, "Float4");
            }

            std::cout << "\n";
            const CsvRunMetadata metadata = make_csv_run_metadata(variant, mode, warmup, passes);
            append_result_to_csv("benchmarks/result.csv", operation, N, result, metadata);
        }
    }

    return 0;
}