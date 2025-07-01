// Implements host launchers for all sin_cos_pow_relu kernel variants.

#include "sin_cos_pow_relu.h"
#include "sin_cos_pow_relu_kernels.cuh"
#include <cuda_runtime.h>
#include "../../cuda_utils.cuh"
#include <stdexcept>
#include <iostream>

// Runs the global memory sin_cos_pow_relu kernel and measures execution time.
extern "C" float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaEvent_t start, stop;
    float ms = -1.0f;

    try {
        std::tie(d_a, d_b, d_c) = allocate_and_copy_to_device(a, b, N);

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        CHECK_CUDA(cudaEventRecord(start));
        sin_cos_pow_relu_global_kernel << <gridSize, blockSize >> > (d_a, d_b, d_c, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));

        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        copy_from_device_and_free(c, d_c, d_a, d_b, N);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_global: " << e.what() << std::endl;
        return -1.0f;
    }
    return ms;
}


// Runs the shared memory sin_cos_pow_relu kernel and measures execution time.
extern "C" float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaEvent_t start, stop;
    float ms = -1.0f;

    try {
        std::tie(d_a, d_b, d_c) = allocate_and_copy_to_device(a, b, N);

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t sharedMemSize = 2 * blockSize * sizeof(float);

        CHECK_CUDA(cudaEventRecord(start));
        sin_cos_pow_relu_shared_kernel << <gridSize, blockSize, sharedMemSize >> > (d_a, d_b, d_c, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));

        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        copy_from_device_and_free(c, d_c, d_a, d_b, N);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_shared: " << e.what() << std::endl;
        return -1.0f;
    }
    return ms;
}

// Runs the float4 vectorized sin_cos_pow_relu kernel and measures execution time.
extern "C" float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N) {
    int N_vec4 = N / 4;

    float4* d_a4 = nullptr;
    float4* d_b4 = nullptr;
    float4* d_c4 = nullptr;
    cudaEvent_t start, stop;
    float ms = -1.0f;

    try {
        std::tie(d_a4, d_b4, d_c4) = allocate_and_copy_to_device_float4(a, b, N_vec4);

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int blockSize = 256;
        int gridSize = (N_vec4 + blockSize - 1) / blockSize;

        CHECK_CUDA(cudaEventRecord(start));
        sin_cos_pow_relu_float4_kernel << <gridSize, blockSize >> > (d_a4, d_b4, d_c4, N_vec4);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));

        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        copy_from_device_and_free_float4(c, d_c4, d_a4, d_b4, N_vec4);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error in run_sin_cos_pow_relu_float4: " << e.what() << std::endl;
        return -1.0f;
    }
    return ms;
}
