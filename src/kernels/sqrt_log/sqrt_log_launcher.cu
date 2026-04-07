/**
 * @file    sqrt_log_launcher.cu
 * @brief   Implements host launchers for all sqrt_log kernel variants (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host-side functions to allocate memory, launch CUDA kernels, measure execution time,
 * and handle data transfers for all sqrt_log kernel variants.
 */


#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "../../../include/cuda_launch_config.h"
#include "../../cuda_utils.cuh"
#include "../../../include/benchmark_utils.h"
#include "sqrt_log.h"
#include "sqrt_log_kernels.cuh"

/**
 * @brief Launches the global memory version of the sqrt_log kernel and measures execution time.
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
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);

        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel([&]() { sqrt_log_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
                                         d_a, d_b, d_c, N);
                                     },
                                     warmup, passes);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_sqrt_log_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

// C++ helpers using PersistentBuffer to avoid repeated allocations/copies
float run_sqrt_log_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
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
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel([&]() { sqrt_log_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N); },
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
            sqrt_log_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N);
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
        std::cerr << "CUDA error in run_sqrt_log_global_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}

float run_sqrt_log_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
        static PersistentBuffer buf_internal;
        if (!buf_internal.initialized || buf_internal.N != N) buf_internal.allocate(N);

        if (mode == BenchmarkMode::EndToEnd)
            buf_internal.copy_to_device(a, b, N);

        float* d_a = buf_internal.d_a;
        float* d_b = buf_internal.d_b;
        float* d_c = buf_internal.d_c;

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        if (mode == BenchmarkMode::KernelOnly)
        {
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel([&]() { sqrt_log_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N); },
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
            sqrt_log_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
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
        std::cerr << "CUDA error in run_sqrt_log_shared_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}

/**
 * @brief Launches the shared-memory version of the sqrt_log kernel and measures execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - An unmeasured warm-up phase (warmup launches) to reduce first-launch overhead and stabilize clocks/caches.
 *  - A timed phase (passes launches) and reports the average time per launch.
 *  - Device-to-host copy of the output and cleanup of device allocations.
 *
 * @param[in]  a       Pointer to the first input array (host).
 * @param[in]  b       Pointer to the second input array (host).
 * @param[out] c       Pointer to the output array (host).
 * @param[in]  N       Number of elements.
 * @param[in]  warmup  Number of warm-up launches (not included in timing).
 * @param[in]  passes  Number of timed launches used to compute the average kernel time.
 * @return             Average kernel execution time in milliseconds, or -1.0f on error.
 */
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        // Read benchmark params from global settings
        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]() {
                sqrt_log_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
            }, warmup, passes);

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
 * The function performs:
 *  - Packs input arrays into float4 vectors (host-side), allocates device memory and copies packed data.
 *  - Launches the vectorized kernel that processes four floats per thread.
 *  - Copies results back to host, unpacks into scalar output, and frees device memory.
 *
 * Requirements:
 *  - The input size N must be divisible by 4; otherwise std::invalid_argument is thrown.
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
        std::cerr << "Error: N must be divisible by 4 for float4 kernel (got " << N << ")\n";
        return -1.0f;
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

        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]()
            {
                sqrt_log_float4_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a4, d_b4, d_c4, N_vec4);
            },
            warmup, passes);

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_sqrt_log_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}