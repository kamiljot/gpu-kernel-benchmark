/**
 * @file    benchmark_dispatch.hpp
 * @brief   Benchmark dispatch for all backend/kernel combinations.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "benchmark/cli_args.hpp"

class CsvLogger;

void dispatch_benchmarks(const Args& args, const std::vector<std::string>& backends,
                         const std::vector<std::string>& kernels, CsvLogger& logger);