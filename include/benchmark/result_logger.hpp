/**
 * @file    result_logger.hpp
 * @brief   Result logging wrapper for benchmark CSV output.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once
#include <string>
#include <vector>

#include "benchmark/benchmark_cpu.hpp"
#include "benchmark/benchmark_cuda.hpp"
#include "utils/csv_logger.hpp"

/**
 * @brief Writes a CPU benchmark result row to CSV.
 */
void log_cpu_result(CsvLogger& logger, const std::string& backend, const std::string& kernel_name, size_t size,
                    int pass, const CpuBenchmarkResult& result);

/**
 * @brief Writes a CUDA benchmark result row to CSV.
 */
void log_cuda_result(CsvLogger& logger, const std::string& backend, const std::string& kernel_name, size_t size,
                     int pass, const CudaBenchmarkResult& result);