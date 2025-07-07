#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

// Structure holding CUDA kernel launch configuration parameters.
struct CudaLaunchConfig
{
    int threads_per_block;  // Number of threads per block
    int blocks_per_grid;    // Number of blocks per grid
};

// Returns the maximum threads per block supported by the current CUDA device.
inline int get_max_threads_per_block()
{
    cudaDeviceProp deviceProp;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get CUDA device.");
    }
    err = cudaGetDeviceProperties(&deviceProp, device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get device properties.");
    }
    return deviceProp.maxThreadsPerBlock;
}

// Selects an optimal number of threads per block, choosing from common sizes
// and ensuring it does not exceed the device maximum.
inline int choose_threads_per_block(int max_threads)
{
    constexpr int preferred_sizes[] = {1024, 512, 256, 128, 64, 32};
    for (int size : preferred_sizes)
    {
        if (size <= max_threads) return size;
    }
    return 32;  // Fallback minimum thread block size
}

// Computes a CUDA launch configuration given the total workload size N.
// Determines appropriate threads per block and blocks per grid.
inline CudaLaunchConfig get_launch_config(int N)
{
    int max_threads = get_max_threads_per_block();
    int threads = choose_threads_per_block(max_threads);
    int blocks = (N + threads - 1) / threads;
    return {threads, blocks};
}