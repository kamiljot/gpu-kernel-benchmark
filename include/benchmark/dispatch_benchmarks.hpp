/**
 * @file    dispatch_benchmarks.hpp
 * @brief   Functions for dispatching and reporting benchmarks for multiple backends/kernels.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once

#include <string>
#include <vector>

#include "benchmark/cli_args.hpp"
#include "utils/csv_logger.hpp"

/**
 * @brief   Runs all requested benchmarks and prints summary.
 * @param   args       Command-line arguments structure.
 * @param   backends   List of backend names to benchmark.
 * @param   kernels    List of kernel names to benchmark.
 * @param   logger     CsvLogger instance to log results.
 */
void dispatch_benchmarks(const Args& args, const std::vector<std::string>& backends,
                         const std::vector<std::string>& kernels, CsvLogger& logger);