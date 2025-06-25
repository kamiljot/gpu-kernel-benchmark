#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include "sin_cos_pow_relu_kernels.cuh"
#include "sin_cos_pow_relu.h"
#include "../../cuda_utils.cuh"

/**
 * @brief Launches the global memory variant of the sin_cos_pow_relu kernel.
 */
float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N) {
    float time_ms = measure_kernel_time([&]() {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        sin_cos_pow_relu_global_kernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, N);
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
        });
    return time_ms;
}

/**
 * @brief Launches the shared memory variant of the sin_cos_pow_relu kernel.
 */
float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N) {
    float time_ms = measure_kernel_time([&]() {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        sin_cos_pow_relu_shared_kernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, N);
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
        });
    return time_ms;
}

/**
 * @brief Launches the float4 vectorized variant of the sin_cos_pow_relu kernel.
 * @throws std::invalid_argument if N is not divisible by 4.
 */
float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N) {
    if (N % 4 != 0) {
        throw std::invalid_argument("Input size N must be divisible by 4 for float4 kernel.");
    }

    int N_vec4 = N / 4;
    float time_ms = measure_kernel_time([&]() {
        auto [d_a4, d_b4, d_c4] = allocate_and_copy_to_device_float4(a, b, N_vec4);
        int blockSize = 256;
        int numBlocks = (N_vec4 + blockSize - 1) / blockSize;
        sin_cos_pow_relu_float4_kernel << <numBlocks, blockSize >> > (d_a4, d_b4, d_c4, N_vec4);
        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
        });
    return time_ms;
}