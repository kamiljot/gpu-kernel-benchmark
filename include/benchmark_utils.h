/**
 * @file    benchmark_utils.h
 * @brief   Input/output utilities for benchmark runs and CSV result logging.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Provides helpers for benchmark parameter management, input loading,
 * and benchmark result persistence.
 */

#pragma once

#include <string>
#include <vector>

/**
 * @brief Benchmark measurement mode.
 *
 * - KernelOnly: measures kernel execution only.
 * - EndToEnd: measures H2D copy + kernel execution + D2H copy.
 */
enum class BenchmarkMode
{
    KernelOnly,
    EndToEnd
};

struct BenchmarkResult;

/**
 * @brief Sets global benchmark parameters used by launchers and helpers.
 *
 * @param[in] warmup Number of unmeasured warm-up launches to perform before timing.
 * @param[in] passes Number of timed measurement passes to average.
 * @param[in] mode   Benchmark measurement mode.
 */
void set_benchmark_params(int warmup, int passes, BenchmarkMode mode);

/**
 * @brief Returns the currently configured warm-up launch count.
 * @return Configured warm-up count.
 */
int get_benchmark_warmup();

/**
 * @brief Returns the currently configured number of measurement passes.
 * @return Configured measurement pass count.
 */
int get_benchmark_passes();

/**
 * @brief Returns the currently configured benchmark measurement mode.
 * @return Configured benchmark mode.
 */
BenchmarkMode get_benchmark_mode();

/**
 * @brief Loads input vectors from a binary file or generates data if the file is missing.
 *
 * The expected file format is:
 * - int32 element count N
 * - N floats for vector a
 * - N floats for vector b
 *
 * If the file does not exist, random input is generated and written to the same path.
 *
 * @param[in]  filename Input file path.
 * @param[out] a        First input vector.
 * @param[out] b        Second input vector.
 * @return True on success, false on read/format failure.
 */
bool read_input_file(const std::string& filename, std::vector<float>& a, std::vector<float>& b);

/**
 * @brief Metadata written to CSV for a single benchmark run.
 */
struct CsvRunMetadata
{
    std::string timestamp_utc;   ///< ISO-8601 UTC timestamp.
    std::string variant;         ///< global/shared/float4/all
    BenchmarkMode mode = BenchmarkMode::KernelOnly; ///< Benchmark mode.
    int warmup = 20;             ///< Warm-up iteration count.
    int passes = 500;            ///< Timed iteration count.
    std::string cuda_arch;       ///< Example: sm_86.
    std::string gpu_name;        ///< Example: NVIDIA GeForce RTX 4060.
    std::string driver_version;  ///< Example: 550.54.
};

/**
 * @brief Builds benchmark run metadata (timestamp + GPU/driver info when available).
 *
 * @param[in] variant Selected kernel variant.
 * @param[in] mode    Benchmark mode.
 * @param[in] warmup  Warm-up iteration count.
 * @param[in] passes  Timed iteration count.
 * @return            Populated metadata structure.
 */
CsvRunMetadata make_csv_run_metadata(const std::string& variant, BenchmarkMode mode, int warmup, int passes);

/**
 * @brief Appends a benchmark result row to CSV with run metadata.
 *
 * Creates a CSV header automatically when the file does not exist or is empty.
 *
 * @param[in] filename  Path to the CSV file.
 * @param[in] operation Benchmark operation name.
 * @param[in] N         Input size.
 * @param[in] result    Benchmark result values.
 * @param[in] metadata  Run metadata written to CSV columns.
 */
void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const BenchmarkResult& result, const CsvRunMetadata& metadata);

/**
 * @brief Backward-compatible benchmark result append (legacy format).
 *
 * This overload writes only:
 * operation, N, cpu_time, gpu_global_time, gpu_shared_time, gpu_float4_time
 *
 * @param[in] filename  Path to the CSV file.
 * @param[in] operation Benchmark operation name.
 * @param[in] N         Input size.
 * @param[in] result    Benchmark result values.
 */
void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const BenchmarkResult& result);
