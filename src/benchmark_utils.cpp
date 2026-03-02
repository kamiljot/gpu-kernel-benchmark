/**
 * @file    benchmark_utils.cpp
 * @brief   Input/output utilities for reading binary input and saving results to CSV.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Implements functions for reading input vectors from a binary file and appending benchmark results to CSV.
 */

#include "benchmark_utils.h"

#include <fstream>
#include <iostream>

#include "input_generator.h"
#include "kernel_dispatch.h"
#include "benchmark_utils.h"

// Default benchmark parameters
static int g_benchmark_warmup = 20;
static int g_benchmark_passes = 500;
static BenchmarkMode g_benchmark_mode = BenchmarkMode::KernelOnly;

/**
 * @brief Sets global benchmark parameters used by launchers and helpers.
 *
 * Stores the desired warm-up launch count, measurement passes and benchmark mode
 * in a process-global location so that launcher implementations can retrieve consistent parameters.
 *
 * @param warmup  Number of unmeasured warm-up launches to perform before timing.
 * @param passes  Number of timed measurement passes to average.
 * @param mode    Benchmark measurement mode (KernelOnly or EndToEnd).
 */
void set_benchmark_params(int warmup, int passes, BenchmarkMode mode)
{
    g_benchmark_warmup = warmup;
    g_benchmark_passes = passes;
    g_benchmark_mode = mode;
}

/**
 * @brief Returns the currently configured warm-up launch count.
 * @return Configured warm-up count.
 */
int get_benchmark_warmup() { return g_benchmark_warmup; }

/**
 * @brief Returns the currently configured number of measurement passes.
 * @return Configured measurement passes count.
 */
int get_benchmark_passes() { return g_benchmark_passes; }

/**
 * @brief Returns the currently configured benchmark measurement mode.
 * @return Configured BenchmarkMode value.
 */
BenchmarkMode get_benchmark_mode() { return g_benchmark_mode; }

/**
 * @brief Loads input vectors a and b from a binary file, or generates random input if file is missing.
 *
 * If the file does not exist, random input is generated and saved to the file.
 *
 * @param filename  Path to the input file.
 * @param a         Vector to be loaded as the first input.
 * @param b         Vector to be loaded as the second input.
 * @return          True if loading (or generation) was successful, false otherwise.
 */
bool read_input_file(const std::string& filename, std::vector<float>& a, std::vector<float>& b)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        // Auto-generate input if file doesn't exist
        std::cout << "Input file not found. Generating random data...\n";
        int N = 1000000;
        generate_random_input(N, a, b);
        write_input_file(filename, a, b);
        return true;
    }

    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(int));
    a.resize(N);
    b.resize(N);
    file.read(reinterpret_cast<char*>(a.data()), N * sizeof(float));
    file.read(reinterpret_cast<char*>(b.data()), N * sizeof(float));
    return file.good();
}

/**
 * @brief Appends one benchmark result to a CSV output file.
 *
 * Each row corresponds to a single benchmark run.
 *
 * @param[in] filename    Path to the CSV file.
 * @param[in] operation   Name of the operation being benchmarked.
 * @param[in] N           Input size.
 * @param[in] result      Benchmark result struct to be appended.
 */
void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const BenchmarkResult& result)
{
    std::ofstream file(filename, std::ios::app);
    file << operation << "," << N << "," << result.cpu_time << "," << result.gpu_global_time << ","
         << result.gpu_shared_time << "," << result.gpu_float4_time << "\n";
}