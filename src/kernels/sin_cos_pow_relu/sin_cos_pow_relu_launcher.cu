// Implements host launchers for all sin_cos_pow_relu kernel variants (global, shared, float4).

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "sin_cos_pow_relu_kernels.cuh"
#include "sin_cos_pow_relu.h"
#include "../../cuda_utils.cuh"
#include "../../../include/cuda_launch_config.h"

extern "C" float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);

        time_ms = launch_kernel_multiple_times([&]() {
            sin_cos_pow_relu_global_kernel << <config.blocks_per_grid, config.threads_per_block >> > (d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

extern "C" float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        time_ms = launch_kernel_multiple_times([&]() {
            sin_cos_pow_relu_shared_kernel << <config.blocks_per_grid, config.threads_per_block, sharedMemSize >> > (d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

extern "C" float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N) {
    if (N % 4 != 0) {
        throw std::invalid_argument("Input size N must be divisible by 4 for float4 kernel.");
    }

    int N_vec4 = N / 4;
    auto h_a4 = pack_and_pad_to_float4(a, N);
    auto h_b4 = pack_and_pad_to_float4(b, N);

    float4* d_a4 = nullptr;
    float4* d_b4 = nullptr;
    float4* d_c4 = nullptr;

    float time_ms = -1.0f;

    try {
        std::tie(d_a4, d_b4, d_c4) = allocate_and_copy_to_device_float4(h_a4.data(), h_b4.data(), N_vec4);

        CudaLaunchConfig config = get_launch_config(N_vec4);

        time_ms = launch_kernel_multiple_times([&]() {
            sin_cos_pow_relu_float4_kernel << <config.blocks_per_grid, config.threads_per_block >> > (d_a4, d_b4, d_c4, N_vec4);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}