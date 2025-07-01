// Implements host launchers for all sqrt_log kernel variants (global, shared, float4).

#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include "sqrt_log_kernels.cuh"
#include "sqrt_log.h"
#include "../../cuda_utils.cuh"

// Launches the global memory version of the sqrt+log kernel
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N) {
    float time_ms = measure_kernel_time([&]() {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        sqrt_log_global_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
        });
    return time_ms;
}

// Launches the shared memory version of the sqrt+log kernel
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N) {
    float time_ms = measure_kernel_time([&]() {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        sqrt_log_shared_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
        });
    return time_ms;
}

// Launches the float4 vectorized version of the sqrt+log kernel
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N) {
    if (N % 4 != 0) {
        throw std::invalid_argument("Input size N must be divisible by 4 for float4 kernel.");
    }

    int N_vec4 = N / 4;
    float time_ms = measure_kernel_time([&]() {
        auto [d_a4, d_b4, d_c4] = allocate_and_copy_to_device_float4(a, b, N_vec4);
        int blockSize = 256;
        int numBlocks = (N_vec4 + blockSize - 1) / blockSize;
        sqrt_log_float4_kernel<<<numBlocks, blockSize>>>(d_a4, d_b4, d_c4, N_vec4);
        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
        });
    return time_ms;
}