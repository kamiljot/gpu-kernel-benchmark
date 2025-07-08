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

/**
 * @brief Loads input vectors a and b from a binary file.
 *
 * If the file does not exist, random input is generated and saved to the file.
 *
 * @param[in]  filename  Path to the input file.
 * @param[out] a         Vector to be loaded as the first input.
 * @param[out] b         Vector to be loaded as the second input.
 * @return     True if loading (or generation) was successful, false otherwise.
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