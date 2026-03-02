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

/**
 * @brief Entry point for single-batch benchmark program.
 *
 * Command line usage:
 *   main [operation] [input_file] [--variant <global|shared|float4|all>]
 *        [--warmup <N>] [--passes <N>] [--mode <kernel|e2e>]
 *
 * The program loads input from a binary file (or generates random input if the file is missing),
 * dispatches the selected operation to CPU and GPU kernels, measures execution times, and prints
 * results to stdout and appends them to a CSV file.
 *
 * @param argc  Argument count.
 * @param argv  Argument values.
 * @return      Exit code (0 on success).
 */
int main(int argc, char* argv[])
{
    std::string operation = "add";
    std::string input_path = "input_file.bin";
    std::string variant = "all";
    int warmup = 20;
    int passes = 500;
    BenchmarkMode mode = BenchmarkMode::KernelOnly;

    if (argc >= 2) operation = argv[1];
    if (argc >= 3) input_path = argv[2];

    // Parse optional --variant, --warmup, --passes, --mode arguments
    for (int i = 3; i < argc; ++i)
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
        else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc)
        {
            passes = std::stoi(argv[i + 1]);
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

    std::vector<float> a, b;
    if (!read_input_file(input_path, a, b))
    {
        std::cout << "Input file not found. Generating random data...\n";
        int N = 1000;
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

    append_result_to_csv("result.csv", operation, static_cast<int>(a.size()), result);
    return 0;
}