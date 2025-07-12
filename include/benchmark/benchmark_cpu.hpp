/**
 * @file    benchmark_cpu.hpp
 * @brief   Standalone function for running CPU benchmarks (modular interface).
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once

#include <string>
#include <vector>

/**
 * @struct CpuBenchmarkResult
 * @brief  Holds the result of a single CPU benchmark pass.
 */
struct CpuBenchmarkResult
{
    double elapsed_ms;  ///< Time elapsed in milliseconds.
};

/**
 * @brief  Runs a CPU kernel benchmark for given inputs.
 * @param  kernel_name Name of the registered kernel.
 * @param  in1         First input vector (host memory).
 * @param  in2         Second input vector (host memory).
 * @param  out         Output vector (host memory, will be overwritten).
 * @param  size        Number of elements to process.
 * @return Struct with elapsed time in ms.
 */
CpuBenchmarkResult run_cpu_benchmark(const std::string& kernel_name, const std::vector<float>& in1,
                                     const std::vector<float>& in2, std::vector<float>& out, size_t size);