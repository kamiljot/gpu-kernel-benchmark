/**
 * @file    benchmark_utils.h
 * @brief   Input/output utilities for benchmark runs and CSV result logging.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Provides functions for loading input data and appending benchmark results to CSV files.
 */

#pragma once
#include <string>
#include <vector>

/**
 * @brief Loads input vectors a and b from a binary file.
 *
 * The binary file is expected to contain the serialized contents of two float vectors.
 *
 * @param[in]  filename  Path to the input file.
 * @param[out] a         Vector to be loaded as the first input.
 * @param[out] b         Vector to be loaded as the second input.
 * @return     True if loading was successful, false otherwise.
 */
bool read_input_file(const std::string& filename, std::vector<float>& a, std::vector<float>& b);

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
                          const struct BenchmarkResult& result);