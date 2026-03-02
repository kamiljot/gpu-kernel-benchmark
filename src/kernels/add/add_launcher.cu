/**
 * @file    add_launcher.cu
 * @brief   Implements host launchers for all add kernel variants (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host-side functions to allocate memory, launch CUDA kernels, measure execution time,
 * and handle data transfers for various addition kernel variants.
 */

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "../../../include/cuda_launch_config.h"
#include "../../cuda_utils.cuh"
#include "../../../include/benchmark_utils.h"
#include "add.h"
#include "add_kernels.cuh"

/**
 * @brief Launches the global memory version of the add kernel and measures execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - A single timed kernel launch that uses global memory for inputs/outputs.
 *  - Device-to-host copy of the output and cleanup of device allocations.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_add_global(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        // For compatibility keep original behavior for extern C API
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        // Read benchmark params from global settings
        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        CudaLaunchConfig config = get_launch_config(N);

        time_ms = benchmark_kernel(
            [&]() { add_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N); },
            warmup, passes);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_add_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}


// C++ helper that uses PersistentBuffer to avoid repeated allocations/copies.
float run_add_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
        // Use an internal persistent buffer to avoid exposing the type in header
        static PersistentBuffer buf_internal;
        if (!buf_internal.initialized || buf_internal.N != N) buf_internal.allocate(N);

        if (mode == BenchmarkMode::EndToEnd)
        {
            buf_internal.copy_to_device(a, b, N);
        }

        float* d_a = buf_internal.d_a;
        float* d_b = buf_internal.d_b;
        float* d_c = buf_internal.d_c;

        CudaLaunchConfig config = get_launch_config(N);

        if (mode == BenchmarkMode::KernelOnly)
        {
            // ensure data present on device
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel([&]() { add_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
                                         d_a, d_b, d_c, N);
                                     }, get_benchmark_warmup(), get_benchmark_passes());
            // copy result back once
            buf_internal.copy_to_host(c);
        }
        else
        {
            // End-to-end: measure H2D + kernel + D2H using CUDA events
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            CHECK_CUDA(cudaEventRecord(start));
            // host->device
            CHECK_CUDA(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
            // kernel
            add_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N);
            // device->host
            CHECK_CUDA(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_add_global_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}

/**
 * @brief Launches the shared memory version of the add kernel and measures execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - Launches the kernel using shared memory (both input arrays are staged into shared memory per block).
 *  - Copies results back to host and frees device allocations.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]() { add_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(
                        d_a, d_b, d_c, N);
            }, warmup, passes);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_add_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the float4 vectorized version of the add kernel and measures execution time.
 *
 * The function performs:
 *  - Packs input arrays into float4 vectors (host-side) and pads if necessary.
 *  - Allocates device memory for packed vectors and copies data to device.
 *  - Launches the vectorized kernel that processes four floats per thread.
 *  - Copies results back to host, unpacks into scalar output, and frees device memory.
 *
 * Requirements:
 *  - The input size N must be divisible by 4; otherwise behavior depends on padding/unpacking.
 *
 * @param[in]  a  Pointer to the first input array (host).
 * @param[in]  b  Pointer to the second input array (host).
 * @param[out] c  Pointer to the output array (host).
 * @param[in]  N  Number of elements (should be divisible by 4 for exact vectorization).
 * @return        Kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N)
{
   
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

        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]() { add_float4_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a4, d_b4, d_c4, N_vec4); },
            warmup, passes);

        // Copy results back to host and free device memory
        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_add_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

/**
 * @brief Launches the shared memory version of the add kernel using a persistent device buffer and measures execution
 * time.
 *
 * The function performs:
 *  - Allocates and reuses device memory for inputs/outputs across launches (persistent buffer).
 *  - Copies input arrays to device memory as needed (depending on BenchmarkMode).
 *  - Launches the shared memory kernel, staging inputs into shared memory per block.
 *  - Measures kernel or end-to-end time using CUDA events.
 *  - Copies results back to host.
 *
 * Requirements:
 *  - The persistent buffer is reused for multiple launches to avoid repeated cudaMalloc/cudaFree.
 *
 * @param[in]  a      Pointer to the first input array (host).
 * @param[in]  b      Pointer to the second input array (host).
 * @param[out] c      Pointer to the output array (host).
 * @param[in]  N      Number of elements.
 * @param[in]  mode   Benchmark measurement mode (KernelOnly or EndToEnd).
 * @return            Kernel execution time in milliseconds, or -1.0f on error.
 */
float run_add_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
        static PersistentBuffer buf_internal;
        if (!buf_internal.initialized || buf_internal.N != N) buf_internal.allocate(N);
        if (mode == BenchmarkMode::EndToEnd) buf_internal.copy_to_device(a, b, N);
        float* d_a = buf_internal.d_a;
        float* d_b = buf_internal.d_b;
        float* d_c = buf_internal.d_c;
        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);
        if (mode == BenchmarkMode::KernelOnly)
        {
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel(
                [&]()
                {
                    add_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b,
                                                                                                           d_c, N);
                },
                get_benchmark_warmup(), get_benchmark_passes());
            buf_internal.copy_to_host(c);
        }
        else
        {
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            CHECK_CUDA(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
            add_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
            CHECK_CUDA(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_add_shared_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}