// Implements host launchers for all sqrt_log kernel variants (global, shared, float4).

#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <iostream>
#include "sqrt_log_kernels.cuh"
#include "sqrt_log.h"
#include "../../cuda_utils.cuh"

// Launches the global memory version of the sqrt+log kernel
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        time_ms = launch_kernel_multiple_times([&]() {
            sqrt_log_global_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sqrt_log_global: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

// Launches the shared memory version of the sqrt+log kernel
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N) {
    float time_ms = -1.0f;

    try {
        auto [d_a, d_b, d_c] = allocate_and_copy_to_device(a, b, N);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t sharedMemSize = 2 * blockSize * sizeof(float);

        time_ms = launch_kernel_multiple_times([&]() {
            sqrt_log_shared_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            }, 1);

        copy_from_device_and_free(c, d_c, d_a, d_b, N);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sqrt_log_shared: " << e.what() << std::endl;
        return -1.0f;
    }

    return time_ms;
}

// Launches the float4 vectorized version of the sqrt+log kernel
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N) {
    int padded_N = (N + 3) / 4 * 4;
    int N_vec4 = padded_N / 4;

    // Pack input arrays to float4 vectors
    auto h_a4 = pack_and_pad_to_float4(a, N);
    auto h_b4 = pack_and_pad_to_float4(b, N);

    float4* d_a4 = nullptr;
    float4* d_b4 = nullptr;
    float4* d_c4 = nullptr;
    float ms = -1.0f;

    try {
        std::tie(d_a4, d_b4, d_c4) = allocate_and_copy_to_device_float4(h_a4.data(), h_b4.data(), N_vec4);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int blockSize = 256;
        int gridSize = (N_vec4 + blockSize - 1) / blockSize;

        CHECK_CUDA(cudaEventRecord(start));
        sqrt_log_float4_kernel << <gridSize, blockSize >> > (d_a4, d_b4, d_c4, N_vec4);
        CHECK_CUDA(cudaEventRecord(stop));

        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sqrt_log_float4: " << e.what() << std::endl;
        return -1.0f;
    }

    return ms;
}