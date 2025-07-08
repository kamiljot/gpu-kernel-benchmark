/**
 * @file    sqrt_log_launcher.cu
 * @brief   Implements host launchers for all sqrt_log kernel variants (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host-side functions to allocate memory, launch CUDA kernels, measure execution time,
 * and handle data transfers for all sqrt_log kernel variants.
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "../../../include/cuda_launch_config.h"
#include "../../cuda_utils.cuh"
#include "sqrt_log.h"
#include "sqrt_log_kernels.cuh"

/**
 * @brief Launches the global memory version of the sqrt_log kernel and measures execution time.
 *
 * Allocates device memory, copies input data, launches the kernel, copies results back, and frees memory.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);

        time_ms = launch_kernel_multiple_times(
            [&]()
            {
                sqrt_log_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_sqrt_log_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the shared memory version of the sqrt_log kernel and measures execution time.
 *
 * Allocates device memory, copies input data, launches the kernel with shared memory, copies results back, and frees
 * memory.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        time_ms = launch_kernel_multiple_times(
            [&]()
            {
                sqrt_log_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b,
                                                                                                            d_c, N);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_sqrt_log_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the float4 vectorized version of the sqrt_log kernel and measures execution time.
 *
 * Packs input arrays into float4 vectors, allocates device memory, copies input data,
 * launches the float4 kernel, copies results back, and frees memory.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements (must be divisible by 4).
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 * @throws        std::invalid_argument if N is not divisible by 4.
 */
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N)
{
    if (N % 4 != 0)
    {
        throw std::invalid_argument("Input size N must be divisible by 4 for float4 kernel.");
    }

    int N_vec4 = N / 4;
    auto h_a4 = pack_and_pad_to_float4(a, N);
    auto h_b4 = pack_and_pad_to_float4(b, N);

    float4* d_a4 = nullptr;
    float4* d_b4 = nullptr;
    float4* d_c4 = nullptr;

    float time_ms = -1.0f;

    try
    {
        std::tie(d_a4, d_b4, d_c4) = allocate_and_copy_to_device_float4(h_a4.data(), h_b4.data(), N_vec4);

        CudaLaunchConfig config = get_launch_config(N_vec4);

        time_ms = launch_kernel_multiple_times(
            [&]()
            {
                sqrt_log_float4_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a4, d_b4, d_c4, N_vec4);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_sqrt_log_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}