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

/**
 * @brief Dispatches and benchmarks selected variant(s) of CPU and GPU kernels for a math operation.
 *
 * Runs the specified operation on CPU and/or selected GPU kernel variants,
 * returns timing results for each variant in a BenchmarkResult struct.
 *
 * @param[in]  operation Name of the operation ("add", "sqrt_log", "sin_cos_pow_relu").
 * @param[in]  a         Pointer to the first input array.
 * @param[in]  b         Pointer to the second input array.
 * @param[out] c         Pointer to the output array.
 * @param[in]  N         Number of elements.
 * @param[in]  variant   Which kernel(s) to run: "global", "shared", "float4", or "all" (default).
 * @return               Struct with timings for all variants.
 * @throws std::invalid_argument on unknown operation.
 */
BenchmarkResult dispatch_and_benchmark(const std::string& operation, const float* a, const float* b, float* c, int N,
                                       const std::string& variant)
{
    BenchmarkResult result;

    if (operation == "add")
    {
        result.cpu_time = run_cpu_add(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_add_global(a, b, c, N);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_add_shared(a, b, c, N);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_add_float4(a, b, c, N);
    }
    else if (operation == "sqrt_log")
    {
        result.cpu_time = run_cpu_sqrt_log(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_sqrt_log_global(a, b, c, N);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_sqrt_log_shared(a, b, c, N);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_sqrt_log_float4(a, b, c, N);
    }
    else if (operation == "sin_cos_pow_relu")
    {
        result.cpu_time = run_cpu_sin_cos_pow_relu(a, b, c, N);
        if (variant == "global" || variant == "all") result.gpu_global_time = run_sin_cos_pow_relu_global(a, b, c, N);
        if (variant == "shared" || variant == "all") result.gpu_shared_time = run_sin_cos_pow_relu_shared(a, b, c, N);
        if (variant == "float4" || variant == "all") result.gpu_float4_time = run_sin_cos_pow_relu_float4(a, b, c, N);
    }
    else
    {
        throw std::invalid_argument("Unknown operation: " + operation);
    }
    return result;
}