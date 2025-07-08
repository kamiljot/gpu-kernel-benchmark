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
 * @brief Launches a global memory CUDA kernel and measures its execution time.
 *
 * @param[in]  d_a   Device pointer to the first input array.
 * @param[in]  d_b   Device pointer to the second input array.
 * @param[out] d_c   Device pointer to the output array.
 * @param[in]  N     Number of elements.
 * @return           Kernel execution time in milliseconds.
 */
float benchmark_global_kernel(const float* d_a, const float* d_b, float* d_c, int N);

/**
 * @brief Launches a shared memory CUDA kernel and measures its execution time.
 *
 * @param[in]  d_a      Device pointer to the first input array.
 * @param[in]  d_b      Device pointer to the second input array.
 * @param[out] d_c      Device pointer to the output array.
 * @param[in]  N        Number of elements.
 * @param[in]  blockSize  Number of threads per block.
 * @return              Kernel execution time in milliseconds.
 */
float benchmark_shared_kernel(const float* d_a, const float* d_b, float* d_c, int N, int blockSize);

/**
 * @brief Launches a float4 CUDA kernel and measures its execution time.
 *
 * @param[in]  a    Device pointer to the first input array.
 * @param[in]  b    Device pointer to the second input array.
 * @param[in]  N    Number of elements.
 * @return          Kernel execution time in milliseconds.
 */
float benchmark_float4_kernel(const float* a, const float* b, int N);