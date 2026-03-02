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
 * @brief Benchmark measurement mode: KernelOnly measures only kernel execution (no H2D/D2H),
 *        EndToEnd measures host->device copy + kernel + device->host copy.
 */
enum class BenchmarkMode { KernelOnly, EndToEnd };

// Setter/getter for global benchmark parameters used by launchers
/**
 * @brief Sets global benchmark parameters used by launchers and helpers.
 *
 * This function stores the desired warm-up launch count, measurement passes and
 * benchmark mode (KernelOnly or EndToEnd) in a process-global location so that
 * launcher implementations can retrieve consistent parameters without changing
 * their external signatures.
 *
 * @param[in] warmup  Number of unmeasured warm-up launches to perform before timing.
 * @param[in] passes  Number of timed measurement passes to average.
 * @param[in] mode    Benchmark measurement mode (KernelOnly or EndToEnd).
 */
void set_benchmark_params(int warmup, int passes, BenchmarkMode mode);

/**
 * @brief Returns the currently configured warm-up launch count.
 *
 * Launchers call this helper to obtain the number of warm-up launches that should
 * be executed prior to timing measurements.
 *
 * @return Configured warm-up count.
 */
int get_benchmark_warmup();

/**
 * @brief Returns the currently configured number of measurement passes.
 *
 * Launchers call this helper to obtain how many timed passes they should run when
 * performing benchmark measurements.
 *
 * @return Configured measurement passes count.
 */
int get_benchmark_passes();

/**
 * @brief Returns the currently configured benchmark measurement mode.
 *
 * The mode controls whether launchers should measure only kernel execution
 * (KernelOnly) or the full host->device + kernel + device->host sequence (EndToEnd).
 *
 * @return Configured BenchmarkMode value.
 */
BenchmarkMode get_benchmark_mode();


/**
 * @brief Loads input vectors a and b from a binary file.
 *
 * The binary file is expected to contain the serialized contents of two float vectors.
 * This helper is used by the benchmark entry points to obtain input data for kernels.
 *
 * @param[in]  filename  Path to the input file.
 * @param[out] a         Vector to be loaded as the first input.
 * @param[out] b         Vector to be loaded as the second input.
 * @return     True if loading was successful, false otherwise.
 */
/**
 * @brief read_input_file launcher (auto-generated comment).
 *
 * The function performs:
 *  - [Describe behavior: device allocation, copies, kernel launch, copies back].
 *
 * @param[in]  filename  const std::string& filename
 * @param[in]  a  std::vector<float>& a
 * @param[in]  b  std::vector<float>& b
 * @return        Kernel execution time in milliseconds.
 */
bool read_input_file(const std::string& filename, std::vector<float>& a, std::vector<float>& b);

/**
 * @brief Appends one benchmark result to a CSV output file.
 *
 * Each row corresponds to a single benchmark run and contains timings for CPU and GPU variants.
 *
 * @param[in] filename    Path to the CSV file.
 * @param[in] operation   Name of the operation being benchmarked.
 * @param[in] N           Input size.
 * @param[in] result      Benchmark result struct to be appended.
 */
void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const struct BenchmarkResult& result);
