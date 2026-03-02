/**
 * @file    gpu_benchmark_utils.h
 * @brief   GPU benchmarking helpers: kernel launches and timing utilities.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains utility functions for launching CUDA kernels and measuring their execution time.
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief Launches and measures a kernel that reads inputs from and writes outputs to global memory.
 *
 * This helper assumes device pointers are already allocated and populated. It is a thin
 * wrapper used by higher-level launchers which manage allocation and host/device copies.
 *
 * @param[in]  d_a   Device pointer to the first input array.
 * @param[in]  d_b   Device pointer to the second input array.
 * @param[out] d_c   Device pointer to the output array.
 * @param[in]  N     Number of elements.
 * @return           Kernel execution time in milliseconds.
 */
float benchmark_global_kernel(const float* d_a, const float* d_b, float* d_c, int N);

/**
 * @brief Launches and measures a shared-memory kernel variant.
 *
 * The function expects device pointers to be valid. It launches the shared-memory
 * variant with the provided block size and measures execution time.
 *
 * @param[in]  d_a       Device pointer to the first input array.
 * @param[in]  d_b       Device pointer to the second input array.
 * @param[out] d_c       Device pointer to the output array.
 * @param[in]  N         Number of elements.
 * @param[in]  blockSize Threads per block (shared memory allocation depends on this).
 * @return              Kernel execution time in milliseconds.
 */
float benchmark_shared_kernel(const float* d_a, const float* d_b, float* d_c, int N, int blockSize);

/**
 * @brief Launches and measures a float4-vectorized kernel.
 *
 * The launcher expects device-side float4 arrays (packed scalars) and returns the
 * kernel execution time measured via CUDA events.
 *
 * @param[in]  a    Device pointer to the first float4 input array.
 * @param[in]  b    Device pointer to the second float4 input array.
 * @param[in]  N    Number of float4 elements.
 * @return          Kernel execution time in milliseconds.
 */
float benchmark_float4_kernel(const float* a, const float* b, int N);
