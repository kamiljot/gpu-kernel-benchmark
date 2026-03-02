/**
 * @file    {{name}}_launcher.cu
 * @brief   Implements host launchers for all {{name}} kernel variants (global, shared, float4).
 * @author  Kamil J.
 * @date    {{date}}
 *
 * Provides host functions to launch all {{name}} kernel variants, measuring their execution time.
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "../../../include/cuda_launch_config.h"
#include "../../cuda_utils.cuh"
#include "{{name}}.h"
#include "{{name}}_kernels.cuh"

/**
 * @brief Launches the global memory version of the {{name}} kernel and measures execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - A single timed kernel launch that uses global memory for inputs/outputs.
 *  - Device-to-host copy of the output and cleanup of device allocations.
 */
extern "C" float run_
{
    {
        name
    }
}
_global(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);

        time_ms = launch_kernel_multiple_times(
            [&]()
            {
                {
                    {
                        name
                    }
                }
                _global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_{{name}}_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the shared memory version of the {{name}} kernel and measures execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - Launches the kernel using shared memory (both input arrays are staged into shared memory per block).
 *  - Copies results back to host and frees device allocations.
 */
extern "C" float run_
{
    {
        name
    }
}
_shared(const float* a, const float* b, float* c, int N)
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
                {
                    {
                        name
                    }
                }
                _shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_{{name}}_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the float4 vectorized version of the {{name}} kernel and measures execution time.
 *
 * The function performs:
 *  - Packs input arrays into float4 vectors (host-side) and pads if necessary.
 *  - Allocates device memory for packed vectors and copies data to device.
 *  - Launches the vectorized kernel that processes four floats per thread.
 *  - Copies results back to host, unpacks into scalar output, and frees device memory.
 *
 * Requirements:
 *  - The input size N should be divisible by 4 for exact vectorization.
 */
extern "C" float run_
{
    {
        name
    }
}
_float4(const float* a, const float* b, float* c, int N)
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
                {
                    {
                        name
                    }
                }
                _float4_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a4, d_b4, d_c4, N_vec4);
                CHECK_CUDA(cudaGetLastError());
            },
            1);

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_{{name}}_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}
