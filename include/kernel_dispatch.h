/**
 * @file    kernel_dispatch.h
 * @brief   Provides kernel dispatch and benchmarking utilities for CPU and GPU kernel variants.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains structures and functions for launching CPU and GPU kernels and measuring their execution times.
 */

#pragma once
#include <string>

/**
 * @brief Holds timing results for CPU and various GPU implementations.
 */
struct BenchmarkResult
{
    float cpu_time = 0.0f;         ///< Execution time for CPU implementation (ms)
    float gpu_global_time = 0.0f;  ///< Execution time for global memory GPU kernel (ms)
    float gpu_shared_time = 0.0f;  ///< Execution time for shared memory GPU kernel (ms)
    float gpu_float4_time = 0.0f;  ///< Execution time for float4 GPU kernel (ms)
};

/**
 * @brief Dispatches the requested operation (e.g., "add", "sqrt_log"), runs CPU and GPU kernels,
 *        and returns their execution times (ms) in a BenchmarkResult struct.
 *
 * @param[in]  operation  Name of the operation ("add", "sqrt_log", etc.).
 * @param[in]  a          Pointer to the first input array.
 * @param[in]  b          Pointer to the second input array.
 * @param[out] c          Pointer to the output array.
 * @param[in]  N          Number of elements.
 * @param[in]  variant    Kernel variant to run ("global", "shared", "float4", or "all"; default is "all").
 * @return                Struct containing timing results for all tested variants.
 */
BenchmarkResult dispatch_and_benchmark(const std::string& operation, const float* a, const float* b, float* c, int N,
                                       const std::string& variant = "all");