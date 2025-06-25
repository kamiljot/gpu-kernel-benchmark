#include "kernel_dispatch.h"
#include "add.h"
#include "sqrt_log.h"
#include "sin_cos_pow_relu.h"
#include "cpu_baseline.h"
#include <stdexcept>
#include <string>

// Dispatch the appropriate kernel variant based on user-provided CLI arguments
BenchmarkResult dispatch_and_benchmark(
    const std::string& operation,
    const std::string& variant,
    const std::string& input_path,
    const float* a,
    const float* b,
    float* c,
    int N
) {
    BenchmarkResult result;

    if (operation == "add") {
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_add_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_add_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_add_float4(a, b, c, N);
        result.cpu_time = run_cpu_add(a, b, c, N);

    }
    else if (operation == "sqrt_log") {
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_sqrt_log_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_sqrt_log_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_sqrt_log_float4(a, b, c, N);
        result.cpu_time = run_cpu_sqrt_log(a, b, c, N);

    }
    else if (operation == "sin_cos_pow_relu") {
        if (variant == "global" || variant == "all")
            result.gpu_global_time = run_sin_cos_pow_relu_global(a, b, c, N);
        if (variant == "shared" || variant == "all")
            result.gpu_shared_time = run_sin_cos_pow_relu_shared(a, b, c, N);
        if (variant == "float4" || variant == "all")
            result.gpu_float4_time = run_sin_cos_pow_relu_float4(a, b, c, N);
        result.cpu_time = run_cpu_sin_cos_pow_relu(a, b, c, N);

    }
    else {
        throw std::invalid_argument("Unsupported operation: " + operation);
    }

    return result;
}
