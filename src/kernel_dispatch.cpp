// Implements dispatch_and_benchmark: runs CPU and all GPU kernel variants for the selected operation.

#include "kernel_dispatch.h"
#include "cpu_baseline.h"
#include "kernels/add/add.h"
#include "kernels/sqrt_log/sqrt_log.h"
#include "kernels/sin_cos_pow_relu/sin_cos_pow_relu.h"
#include <stdexcept>


// Dispatches and benchmarks selected variant(s) of CPU and GPU kernels for operation.
BenchmarkResult dispatch_and_benchmark(const std::string& operation,
    const float* a, const float* b, float* c, int N,
    const std::string& variant) {
    BenchmarkResult result;

    if (operation == "add") {
        result.cpu_time = run_cpu_add(a, b, c, N);
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_add_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_add_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_add_float4(a, b, c, N);
    }
    else if (operation == "sqrt_log") {
        result.cpu_time = run_cpu_sqrt_log(a, b, c, N);
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_sqrt_log_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_sqrt_log_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_sqrt_log_float4(a, b, c, N);
    }
    else if (operation == "sin_cos_pow_relu") {
        result.cpu_time = run_cpu_sin_cos_pow_relu(a, b, c, N);
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_sin_cos_pow_relu_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_sin_cos_pow_relu_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_sin_cos_pow_relu_float4(a, b, c, N);
    }
    else {
        throw std::invalid_argument("Unknown operation: " + operation);
    }
    return result;
}