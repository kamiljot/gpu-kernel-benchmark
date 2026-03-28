/**
 * @file    {{name}}_launcher.cu
 * @brief   Implements host launchers for all {{name}} kernel variants (global, shared, float4).
 * @author  Kamil J.
 * @date    {{date}}
 *
 * Provides host functions to launch all {{name}} kernel variants, measuring their execution time.
 */

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "../../../include/cuda_launch_config.h"
#include "../../../include/benchmark_utils.h"
#include "../../cuda_utils.cuh"
#include "{{name}}.h"
#include "{{name}}_kernels.cuh"

/**
 * @brief Launches the global memory version of the {{name}} kernel and measures execution time.
 */
extern "C" float run_{{name}}_global(const float* a, const float* b, float* c, int N)
{
    float time_ms = -1.0f;

    try
    {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        CudaLaunchConfig config = get_launch_config(N);
        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]() { {{name}}_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N); },
            warmup, passes);

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
 * @brief Launches the global memory {{name}} kernel using a persistent device buffer.
 */
float run_{{name}}_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
        static PersistentBuffer buf_internal;
        if (!buf_internal.initialized || buf_internal.N != N) buf_internal.allocate(N);

        float* d_a = buf_internal.d_a;
        float* d_b = buf_internal.d_b;
        float* d_c = buf_internal.d_c;

        CudaLaunchConfig config = get_launch_config(N);

        if (mode == BenchmarkMode::KernelOnly)
        {
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel(
                [&]() { {{name}}_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N); },
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
            {{name}}_global_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a, d_b, d_c, N);
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
        std::cerr << "CUDA error in run_{{name}}_global_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}

/**
 * @brief Launches the shared memory version of the {{name}} kernel and measures execution time.
 */
extern "C" float run_{{name}}_shared(const float* a, const float* b, float* c, int N)
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
            [&]() {
                {{name}}_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
            }, warmup, passes);

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
 * @brief Launches the shared memory {{name}} kernel using a persistent device buffer.
 */
float run_{{name}}_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode)
{
    float time_ms = -1.0f;
    try
    {
        static PersistentBuffer buf_internal;
        if (!buf_internal.initialized || buf_internal.N != N) buf_internal.allocate(N);

        float* d_a = buf_internal.d_a;
        float* d_b = buf_internal.d_b;
        float* d_c = buf_internal.d_c;

        CudaLaunchConfig config = get_launch_config(N);
        size_t sharedMemSize = 2 * config.threads_per_block * sizeof(float);

        if (mode == BenchmarkMode::KernelOnly)
        {
            buf_internal.copy_to_device(a, b, N);
            time_ms = benchmark_kernel(
                [&]() {
                    {{name}}_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
                }, get_benchmark_warmup(), get_benchmark_passes());
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
            {{name}}_shared_kernel<<<config.blocks_per_grid, config.threads_per_block, sharedMemSize>>>(d_a, d_b, d_c, N);
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
        std::cerr << "CUDA error in run_{{name}}_shared_with_buffer: " << e.what() << std::endl;
        return -1.0f;
    }
    return time_ms;
}

/**
 * @brief Launches the float4 vectorized version of the {{name}} kernel and measures execution time.
 */
extern "C" float run_{{name}}_float4(const float* a, const float* b, float* c, int N)
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
        int warmup = get_benchmark_warmup();
        int passes = get_benchmark_passes();

        time_ms = benchmark_kernel(
            [&]() { {{name}}_float4_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(d_a4, d_b4, d_c4, N_vec4); },
            warmup, passes);

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error in run_{{name}}_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}