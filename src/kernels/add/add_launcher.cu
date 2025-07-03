// Implements host launchers for all add kernel variants (global, shared, float4).

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "add_kernels.cuh"
#include "add.h"
#include "../../cuda_utils.cuh"

// Launches the global memory version of the add kernel and measures execution time
extern "C" float run_add_global(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        // Allocate device memory and copy input data
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // Launch kernel and measure execution time
        time_ms = launch_kernel_multiple_times([&]() {
            add_global_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        // Copy results back and free device memory
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_add_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

// Launches the shared memory version of the add kernel and measures execution time
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        // Allocate device memory and copy input data
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t sharedMemSize = 2 * blockSize * sizeof(float); // For two arrays in shared memory

        // Launch kernel and measure execution time
        time_ms = launch_kernel_multiple_times([&]() {
            add_shared_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        // Copy results back and free device memory
        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_add_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

// Launches the float4 vectorized version of the add kernel and measures execution time
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N) {
    if (N % 4 != 0) {
        throw std::invalid_argument("Input size N must be divisible by 4 for float4 kernel.");
    }
    int N_vec4 = N / 4;

    // Pack input float arrays into float4 vectors with padding.
    auto h_a4 = pack_and_pad_to_float4(a, N);
    auto h_b4 = pack_and_pad_to_float4(b, N);

    float4* d_a4 = nullptr;
    float4* d_b4 = nullptr;
    float4* d_c4 = nullptr;

    float time_ms = -1.0f;

    try {
        // Allocate device memory and copy input data
        std::tie(d_a4, d_b4, d_c4) = allocate_and_copy_to_device_float4(h_a4.data(), h_b4.data(), N_vec4);

        int blockSize = 256;
        int gridSize = (N_vec4 + blockSize - 1) / blockSize;

        // Measure kernel execution time
        time_ms = launch_kernel_multiple_times([&]() {
            add_float4_kernel << <gridSize, blockSize >> > (d_a4, d_b4, d_c4, N_vec4);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        // Copy results back to host and free device memory
        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_add_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}