/**
 * @file    benchmark_utils.cpp
 * @brief   Input/output utilities for reading benchmark input and writing CSV output.
 * @author  Kamil J.
 * @date    2025-07-07
 */

#include "benchmark_utils.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

#include "input_generator.h"
#include "kernel_dispatch.h"

// Default benchmark parameters
static int g_benchmark_warmup = 20;
static int g_benchmark_passes = 500;
static BenchmarkMode g_benchmark_mode = BenchmarkMode::KernelOnly;

/**
 * @brief Converts benchmark mode to CSV string.
 * @param[in] mode Benchmark mode.
 * @return "kernel" for KernelOnly, "e2e" for EndToEnd.
 */
static std::string benchmark_mode_to_string(BenchmarkMode mode)
{
    return mode == BenchmarkMode::KernelOnly ? "kernel" : "e2e";
}

/**
 * @brief Returns current UTC timestamp in ISO-8601 format.
 * @return Timestamp string, e.g. "2026-03-02T18:22:10Z".
 */
static std::string utc_now_iso8601()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t now_time = clock::to_time_t(now);

    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

/**
 * @brief Checks whether CSV header should be written.
 * @param[in] filename CSV file path.
 * @return True if file does not exist or is empty.
 */
static bool should_write_csv_header(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.good())
    {
        return true;
    }

    return in.peek() == std::ifstream::traits_type::eof();
}

/**
 * @brief Ensures that the parent directory of a file path exists, creating it if needed.
 * @param[in] filepath Path to the file whose parent directory should exist.
 */
static void ensure_parent_directory_exists(const std::string& filepath)
{
    std::filesystem::path p(filepath);
    if (p.has_parent_path())
    {
        std::filesystem::create_directories(p.parent_path());
    }
}

void set_benchmark_params(int warmup, int passes, BenchmarkMode mode)
{
    g_benchmark_warmup = warmup;
    g_benchmark_passes = passes;
    g_benchmark_mode = mode;
}

int get_benchmark_warmup()
{
    return g_benchmark_warmup;
}

int get_benchmark_passes()
{
    return g_benchmark_passes;
}

BenchmarkMode get_benchmark_mode()
{
    return g_benchmark_mode;
}

bool read_input_file(const std::string& filename, std::vector<float>& a, std::vector<float>& b)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        return false;
    }

    int N = 0;
    file.read(reinterpret_cast<char*>(&N), sizeof(int));
    if (!file.good() || N <= 0)
    {
        return false;
    }

    a.resize(static_cast<size_t>(N));
    b.resize(static_cast<size_t>(N));
    file.read(reinterpret_cast<char*>(a.data()), static_cast<std::streamsize>(N) * sizeof(float));
    file.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(N) * sizeof(float));

    return file.good();
}

CsvRunMetadata make_csv_run_metadata(const std::string& variant, BenchmarkMode mode, int warmup, int passes)
{
    CsvRunMetadata metadata{};
    metadata.timestamp_utc = utc_now_iso8601();
    metadata.variant = variant;
    metadata.mode = mode;
    metadata.warmup = warmup;
    metadata.passes = passes;
    metadata.cuda_arch = "unknown";
    metadata.gpu_name = "unknown";
    metadata.driver_version = "unknown";

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return metadata;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
    {
        metadata.gpu_name = prop.name;
        metadata.cuda_arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
    }

    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version > 0)
    {
        const int major = driver_version / 1000;
        const int minor = (driver_version % 1000) / 10;
        metadata.driver_version = std::to_string(major) + "." + std::to_string(minor);
    }

    return metadata;
}

void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const BenchmarkResult& result, const CsvRunMetadata& metadata)
{
    ensure_parent_directory_exists(filename);
    const bool write_header = should_write_csv_header(filename);

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open CSV file: " << filename << "\n";
        return;
    }

    if (write_header)
    {
        file << "timestamp,operation,variant,N,mode,warmup,passes,cuda_arch,gpu_name,driver_version,"
             << "cpu_ms,gpu_global_ms,gpu_shared_ms,gpu_float4_ms\n";
    }

    file << metadata.timestamp_utc << ","
         << operation << ","
         << metadata.variant << ","
         << N << ","
         << benchmark_mode_to_string(metadata.mode) << ","
         << metadata.warmup << ","
         << metadata.passes << ","
         << metadata.cuda_arch << ","
         << metadata.gpu_name << ","
         << metadata.driver_version << ","
         << result.cpu_time << ","
         << result.gpu_global_time << ","
         << result.gpu_shared_time << ","
         << result.gpu_float4_time << "\n";
}

void append_result_to_csv(const std::string& filename, const std::string& operation, int N,
                          const BenchmarkResult& result)
{
    ensure_parent_directory_exists(filename);

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open CSV file: " << filename << "\n";
        return;
    }

    file << operation << ","
         << N << ","
         << result.cpu_time << ","
         << result.gpu_global_time << ","
         << result.gpu_shared_time << ","
         << result.gpu_float4_time << "\n";
}