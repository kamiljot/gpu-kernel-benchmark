/**
 * @file    result_logger.cpp
 * @brief   Result logging wrapper for benchmark CSV output.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#include "benchmark/result_logger.hpp"

void log_cpu_result(CsvLogger& logger, const std::string& backend, const std::string& kernel_name, size_t size,
                    int pass, const CpuBenchmarkResult& result)
{
    logger.write_row({backend, kernel_name, std::to_string(size), std::to_string(pass),
                      std::to_string(result.elapsed_ms), "", "", ""});
}

void log_cuda_result(CsvLogger& logger, const std::string& backend, const std::string& kernel_name, size_t size,
                     int pass, const CudaBenchmarkResult& result)
{
    logger.write_row({backend, kernel_name, std::to_string(size), std::to_string(pass), "",
                      std::to_string(result.transfer_in_ms), std::to_string(result.kernel_ms),
                      std::to_string(result.transfer_out_ms)});
}