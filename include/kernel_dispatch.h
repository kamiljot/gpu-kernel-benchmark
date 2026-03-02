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
#include "benchmark_utils.h"

/**
 * @brief Holds timing results for CPU and various GPU kernel variants.
 *
 * This struct is returned by dispatch_and_benchmark and contains measured execution times
 * for the CPU reference implementation and each GPU kernel variant (global, shared, float4).
 * All times are in milliseconds.
 */
struct BenchmarkResult {
    float cpu_time = 0.0f;         ///< Execution time for CPU implementation (ms)
    float gpu_global_time = 0.0f;  ///< Execution time for global memory GPU kernel (ms)
    float gpu_shared_time = 0.0f;  ///< Execution time for shared memory GPU kernel (ms)
    float gpu_float4_time = 0.0f;  ///< Execution time for float4 GPU kernel (ms)
};

/**
 * @brief Dispatches the requested operation, runs CPU and GPU kernels, and returns timing results.
 *
 * This function selects the appropriate CPU and GPU kernel launchers for the requested operation
 * (e.g., "add", "sqrt_log", "sin_cos_pow_relu") and runs them on the provided input arrays.
 * It measures execution times for each variant and returns them in a BenchmarkResult struct.
 *
 * @param[in]  operation  Name of the operation ("add", "sqrt_log", etc.).
 * @param[in]  a          Pointer to the first input array.
 * @param[in]  b          Pointer to the second input array.
 * @param[out] c          Pointer to the output array.
 * @param[in]  N          Number of elements.
 * @param[in]  variant    Kernel variant to run ("global", "shared", "float4", or "all"; default is "all").
 * @param[in]  warmup     Number of warm-up launches before timing (default: 20).
 * @param[in]  passes     Number of timed measurement passes (default: 500).
 * @param[in]  mode       Benchmark measurement mode (KernelOnly or EndToEnd; default: KernelOnly).
 * @return                Struct containing timing results for all tested variants.
 */
enum class BenchmarkMode;
BenchmarkResult dispatch_and_benchmark(const std::string& operation, const float* a, const float* b, float* c, int N,
                                       const std::string& variant = "all", int warmup = 20, int passes = 500,
                                       BenchmarkMode mode = BenchmarkMode::KernelOnly);
