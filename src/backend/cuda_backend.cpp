/**
 * @file    cuda_backend.cpp
 * @brief   CUDA backend implementation for GPU kernel benchmarking framework.
 * @author  Kamil J.
 * @date    2025-07-11
 */

//#ifdef USE_CUDA

#include "backend/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

// Example: extern "C" kernel launchers
extern "C" void launch_cuda_add_global(float*, float*, float*, size_t);

std::string CudaBackend::name() const
{
    return "cuda";
}

float* CudaBackend::allocate(size_t num_elements)
{
    float* ptr = nullptr;
    cudaMalloc(&ptr, num_elements * sizeof(float));
    return ptr;
}

void CudaBackend::free(float* ptr)
{
    cudaFree(ptr);
}

void CudaBackend::copy_to_device(float* dst, const float* src, size_t num_elements)
{
    cudaMemcpy(dst, src, num_elements * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaBackend::copy_to_host(float* dst, const float* src, size_t num_elements)
{
    cudaMemcpy(dst, src, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaBackend::measure_transfer_in(float* d_in1, float* d_in2, const float* h_in1, const float* h_in2, size_t size,
                                      float& elapsed_ms)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(d_in1, h_in1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaBackend::measure_kernel(const std::string& kernel_name, float* d_in1, float* d_in2, float* d_out, size_t size,
                                 float& elapsed_ms)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (kernel_name == "add_global")
    {
        launch_cuda_add_global(d_in1, d_in2, d_out, size);
    }
    // More kernel launchers here...

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaBackend::measure_transfer_out(float* h_out, float* d_out, size_t size, float& elapsed_ms)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaBackend::launch_kernel(const std::string& kernel_name, float* d_in1, float* d_in2, float* d_out,
                                const float* h_in1, const float* h_in2, float* h_out, size_t size, GpuTiming* timing)
{
    float t_in = 0, t_kernel = 0, t_out = 0;
    measure_transfer_in(d_in1, d_in2, h_in1, h_in2, size, t_in);
    measure_kernel(kernel_name, d_in1, d_in2, d_out, size, t_kernel);
    measure_transfer_out(h_out, d_out, size, t_out);

    if (timing)
    {
        timing->transfer_in_ms = t_in;
        timing->kernel_ms = t_kernel;
        timing->transfer_out_ms = t_out;
    }
}

//#endif  // USE_CUDA