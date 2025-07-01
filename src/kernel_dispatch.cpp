#include "kernel_dispatch.h"
#include "cpu_baseline.h"
#include "kernels/add/add.h"
#include "kernels/sqrt_log/sqrt_log.h"
#include <stdexcept>

BenchmarkResult dispatch_and_benchmark(const std::string& operation,
    const float* a, const float* b, float* c, int N) {
    BenchmarkResult result;

    if (operation == "add") {
        result.cpu_time = run_cpu_add(a, b, c, N);
        result.gpu_global_time = run_add_global(a, b, c, N);
        result.gpu_shared_time = run_add_shared(a, b, c, N);
        result.gpu_float4_time = run_add_float4(a, b, c, N);
    }
    else if (operation == "sqrt_log") {
        result.cpu_time = run_cpu_sqrt_log(a, b, c, N);
        result.gpu_global_time = run_sqrt_log_global(a, b, c, N);
        result.gpu_shared_time = run_sqrt_log_shared(a, b, c, N);
        result.gpu_float4_time = run_sqrt_log_float4(a, b, c, N);
    }
    else {
        throw std::invalid_argument("Unknown operation: " + operation);
    }

    return result;
}
