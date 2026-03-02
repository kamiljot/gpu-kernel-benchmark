/**
 * @file    kernel_dispatch.cpp
 * @brief   Implements dispatch_and_benchmark: runs CPU and all GPU kernel variants for the selected operation.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Provides the dispatcher for launching CPU and all GPU kernel variants for the selected math operation.
 */

#include "kernel_dispatch.h"

#include <stdexcept>

#include "cpu_baseline.h"
#include "kernels/add/add.h"
#include "kernels/sin_cos_pow_relu/sin_cos_pow_relu.h"
#include "kernels/sqrt_log/sqrt_log.h"
#include "benchmark_utils.h"

/**
 * @brief Dispatches and benchmarks selected variant(s) of CPU and GPU kernels for a math operation.
 *
 * This function selects the appropriate CPU and GPU kernel launchers for the requested operation
 * (e.g., "add", "sqrt_log", "sin_cos_pow_relu") and runs them on the provided input arrays.
 * It measures execution times for each variant and returns them in a BenchmarkResult struct.
 *
 * The function sets global benchmark parameters (warmup, passes, mode) which are read by
 * persistent-buffer launchers to ensure consistent measurement across all kernel variants.
 *
 * @param operation Name of the operation ("add", "sqrt_log", "sin_cos_pow_relu").
 * @param a         Pointer to the first input array.
 * @param b         Pointer to the second input array.
 * @param c         Pointer to the output array.
 * @param N         Number of elements.
 * @param variant   Which kernel(s) to run: "global", "shared", "float4", or "all" (default).
 * @param warmup    Number of warm-up launches before timing (default: 20).
 * @param passes    Number of timed measurement passes (default: 500).
 * @param mode      Benchmark measurement mode (KernelOnly or EndToEnd; default: KernelOnly).
 * @return          Struct with timings for all variants.
 * @throws std::invalid_argument on unknown operation.
 */
BenchmarkResult dispatch_and_benchmark(const std::string& operation, const float* a, const float* b, float* c, int N,
                                       const std::string& variant, int warmup, int passes,
                                       BenchmarkMode mode)
{
    BenchmarkResult result;
    // Set global benchmark params for launchers that read them
    set_benchmark_params(warmup, passes, mode);

    if (operation == "add")
    {
        result.cpu_time = run_cpu_add(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_add_global_with_buffer(a, b, c, N, mode);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_add_shared_with_buffer(a, b, c, N, mode);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_add_float4(a, b, c, N);
    }
    else if (operation == "sqrt_log")
    {
        result.cpu_time = run_cpu_sqrt_log(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_sqrt_log_global_with_buffer(a, b, c, N, mode);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_sqrt_log_shared_with_buffer(a, b, c, N, mode);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_sqrt_log_float4(a, b, c, N);
    }
    else if (operation == "sin_cos_pow_relu")
    {
        result.cpu_time = run_cpu_sin_cos_pow_relu(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_sin_cos_pow_relu_global_with_buffer(a, b, c, N, mode);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_sin_cos_pow_relu_shared_with_buffer(a, b, c, N, mode);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_sin_cos_pow_relu_float4(a, b, c, N);
    }
    else
    {
        throw std::invalid_argument("Unknown operation: " + operation);
    }
    return result;
}